//! Execution-state and call-frame machinery shared by the deterministic run
//! front doors.
//!
//! `ExecutionState` owns every mutable runtime value: the heap, globals, the
//! scheduler, local-client sessions, and type metadata. `CallFrame` is the
//! hot per-procedure state the interpreter advances, with rare argument,
//! exception, and diagnostic state pushed behind `CallFrameCold` so ordinary
//! frames stay small.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use crate::builtins;
use crate::builtins::{
    advance_native_walks, execute_external_call, execute_list_binary_operator,
    execute_list_compound_operator, execute_list_method, execute_output, execute_regex_method,
    execute_standard_builtin, execute_standard_builtin_with_usr, is_regex_datum, is_subtype,
};
use crate::bytecode::{
    CompoundAssignmentOperator, CompoundListIndexOperator, InstanceInitializer, Instruction,
    ListEntryKind, Module, ProcedureId, Program, TypePredicateKind,
};
use crate::compile::{EXPANDED_ARGUMENT_COUNT, to_local_index};
use crate::tgm_planner;
use crate::value_ops::{
    ExecutionContext, allocate_dm_array, allocate_matrix, allocate_or_replace_engine_datum,
    allocate_vector, apply_icon_blend, apply_icon_map_colors, apply_icon_set_intensity,
    area_coordinate_field, assign_datum_or_shared_field, atom_contents_iteration_snapshot,
    block_builtin, builtin_length, canonicalize_owned_value, canonicalize_value, compare_values,
    construct_matrix, construct_sized_list, construct_vector, constructor_target_if_present,
    copy_text_builtin, datum_field_or_shared, datum_field_requires_special_read,
    deterministic_unit, direction_towards_builtin, dm_list_length_number, dm_list_resize_length,
    dynamic_call_target, dynamic_call_target_named_at_callsite, engine_root_initial_field_maps,
    engine_root_paths, execute_animate, execute_del, execute_icon_method, execute_matrix_binary,
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
    DmmMeasurement, ExceptionHandler, ExecutionLimits, GlobalStore, HeapReference,
    LocalClientAppearance, LocalClientError, LocalClientMapSnapshot, LocalClientMapTile,
    LocalClientPromptKind, LocalClientPromptResponse, LocalClientScreenAppearance,
    LocalClientState, LocalClientUiEvent, LocalMovementDirection, LocalScreenPointerEvent,
    MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES, MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE,
    NativeWalk, ParsedDmm, PendingLocalPrompt, PendingPromptContinuation, PendingVerbInvocation,
    QuiescentHeapCompaction, RuntimeError, STARTUP_INSTRUCTION_CATEGORY_COUNT, SavefileState,
    ScheduledSpawn, ShuttleTracePostReturn, SimpleIterationValue, TgmDrive, TgmProfile,
    VerbParameterType, advance_headless_world_clock, allocate_initialized_datum,
    assign_datum_field, atoms_profile_enabled, atoms_profile_snapshot_lines_if_due,
    boot_dashboard_enabled, boot_trace_enabled, canonical_istext, canonical_static_native_builtin,
    canonical_tgm_load_path, canonical_type2parent, canonical_type2parent_target,
    datum_field_or_initial, datum_shared_storage, drive_ruin_candidate_scan, drive_tgm_load,
    dynamic_call_target_named, emit_atoms_profile, emit_tgm_profile, engine_builtin_initial_fields,
    engine_builtin_initial_value, engine_root_initial_value, execute_compact_fast_instruction,
    false_tick_check_target, initialize_existing_datum, instance_initializer_plan,
    is_atom_type_path, is_atoms_initialize_path, is_subsystem_initialize_path,
    lazy_atom_list_field, local_prompt_spec, mark_boot_trace_frame, numeric_dispatch_candidate,
    parse_heap_reference, prepare_iteration_consumes_fresh_block, queue_next_verb_prompt,
    register_prompt, set_world_numeric_field, shuttle_trace_emit_snapshot, shuttle_trace_enabled,
    shuttle_trace_prepare_call, shuttle_trace_slot_from_arguments,
    simple_iteration_field_assignment, startup_instruction_category,
    startup_instruction_profile_enabled, startup_profile_enabled, tgm_profiling_enabled,
    trace_tgm_route, try_run_build_coordinate_prefix, try_run_camera_chunk_fast_path,
    try_run_discover_offset_fast_path, try_run_dmm_preload_measurement_fast_path,
    try_run_guarded_jit, try_run_numeric_dispatch_block, try_run_numeric_local_update,
    try_run_numeric_loop_branch, try_run_parsed_dmm_new_fast_path,
    try_run_register_signal_fast_path, try_run_rooted_list_jit, try_run_ruin_affected_turfs_batch,
    try_run_tgm_build_cache_simple_member, world_numeric_field,
};
#[cfg(test)]
use dm_core::SourceSpan;
use dm_dmf::{ClientSession, ControlTree, DiagnosticSeverity, UiEvent, parse as parse_dmf};
use dm_jit::{NumericExecutionState, NumericRunOutcome};
use dm_value::{
    DatumId, FieldName, ListId, ModifiedTypePath, PackedValue, TypePath, Value, ValueError,
    ValueHeap,
};
use smallvec::SmallVec;

pub(crate) fn build_type_intervals(
    parents: &BTreeMap<TypePath, Option<TypePath>>,
) -> BTreeMap<TypePath, (u32, u32)> {
    fn visit(
        node: &TypePath,
        children: &BTreeMap<TypePath, Vec<TypePath>>,
        seen: &mut HashSet<TypePath>,
        intervals: &mut BTreeMap<TypePath, (u32, u32)>,
        clock: &mut u32,
    ) {
        if !seen.insert(node.clone()) {
            return;
        }
        let start = *clock;
        *clock = clock.saturating_add(1);
        if let Some(descendants) = children.get(node) {
            for child in descendants {
                visit(child, children, seen, intervals, clock);
            }
        }
        let end = *clock;
        *clock = clock.saturating_add(1);
        intervals.insert(node.clone(), (start, end));
    }

    let mut children = BTreeMap::<TypePath, Vec<TypePath>>::new();
    let mut roots = Vec::new();
    for (path, parent) in parents {
        if let Some(parent) = parent
            && parents.contains_key(parent)
        {
            children
                .entry(parent.clone())
                .or_default()
                .push(path.clone());
        } else {
            roots.push(path.clone());
        }
    }
    let mut seen = HashSet::new();
    let mut intervals = BTreeMap::new();
    let mut clock = 0u32;
    for root in roots {
        visit(&root, &children, &mut seen, &mut intervals, &mut clock);
    }
    // Invalid or cyclic catalogs should retain deterministic behavior instead
    // of silently dropping types from the acceleration index.
    for path in parents.keys() {
        visit(path, &children, &mut seen, &mut intervals, &mut clock);
    }
    intervals
}
/// Mutable heap state shared by executions in one runtime world.
///
/// Values contain only stable logical handles. All mutable list and datum
/// storage remains here so aliases across calls resolve to one identity.
pub struct ExecutionState {
    // Live DM state has one authoritative mutation thread. This identity is
    // process-local runtime state and is deliberately absent from snapshots.
    pub(crate) owner_thread: OnceLock<ThreadId>,
    pub(crate) heap: ValueHeap,
    pub(crate) associative_lists: HashSet<ListId>,
    pub(crate) reference_lists: HashSet<ListId>,
    pub(crate) savefiles: HashMap<DatumId, SavefileState>,
    pub(crate) savefile_entries: HashMap<DatumId, (DatumId, String)>,
    pub(crate) client_sessions: BTreeMap<DatumId, ClientSession>,
    pub(crate) interactive_local_clients: HashSet<DatumId>,
    pub(crate) local_client_skin: Option<ControlTree>,
    pub(crate) local_client_outbound_events: BTreeMap<DatumId, Vec<LocalClientUiEvent>>,
    pub(crate) local_client_mobs: BTreeMap<DatumId, DatumId>,
    pub(crate) local_client_commands: Vec<(u64, DatumId, LocalMovementDirection)>,
    pub(crate) local_client_command_sequence: u64,
    pub(crate) local_guest_sequence: u64,
    pub(crate) local_prompt_sequence: u64,
    pub(crate) pending_local_prompts: BTreeMap<u64, PendingLocalPrompt>,
    pub(crate) global_vars_proxy: Option<ListId>,
    pub(crate) datum_vars_proxies: HashMap<ListId, DatumId>,
    pub(crate) datum_vars_by_datum: HashMap<DatumId, ListId>,
    pub(crate) contents_owners: HashMap<ListId, DatumId>,
    pub(crate) vis_contents_owners: HashMap<ListId, DatumId>,
    pub(crate) vis_locs_owners: HashMap<ListId, DatumId>,
    pub(crate) shared_fields: Arc<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>>,
    pub(crate) instance_initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
    // Runtime `new` repeatedly flattens the same immutable parent initializer
    // chain. Retain one ordered plan per concrete type; individual initializer
    // values are still cloned/applied for each datum.
    pub(crate) instance_initializer_plans: HashMap<TypePath, Arc<[InstanceInitializer]>>,
    pub(crate) initial_prototypes: BTreeMap<TypePath, DatumId>,
    pub(crate) initial_prototypes_initializing: HashSet<TypePath>,
    pub(crate) instance_initializer_module: Option<Arc<Module>>,
    pub(crate) globals: GlobalStore,
    pub(crate) initial_globals: BTreeMap<FieldName, Value>,
    pub(crate) type_paths: Arc<std::collections::BTreeSet<TypePath>>,
    pub(crate) type_parents: Arc<BTreeMap<TypePath, Option<TypePath>>>,
    pub(crate) type_intervals: Arc<BTreeMap<TypePath, (u32, u32)>>,
    // Runtime type metadata and a compiled module's procedure table are immutable
    // while DM executes. Receiver dispatch can therefore retain both successful
    // and failed parent-chain resolutions. The cache is cleared only by the host
    // APIs that replace the type-parent catalog.
    pub(crate) dynamic_receiver_targets:
        HashMap<(u64, TypePath), HashMap<String, Option<ProcedureId>>>,
    // Static member-call sites overwhelmingly see the same concrete receiver
    // types during atom initialization. Cache their resolved target without
    // re-hashing full type-path and selector strings on every invocation. The
    // retained TypePath keeps the process-local storage identity collision-free.
    pub(crate) dynamic_callsite_targets:
        HashMap<(u64, ProcedureId, usize, usize), (TypePath, ProcedureId)>,
    // Static field-read sites can retain a dense datum slot. Every hit still
    // validates both the field name and current layout, and engine-special
    // fields never enter this path.
    pub(crate) declared_field_slots: HashMap<(u64, ProcedureId, u16), u16>,
    pub(crate) declared_field_quickening: DeclaredFieldQuickeningMetrics,
    pub(crate) initial_values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,
    // The initial-value catalog is immutable but can contain millions of
    // scalar values. Derive the only heap handles it can retain when the
    // catalog is installed, so every collection need only append this compact
    // deduplicated snapshot instead of walking the complete catalog.
    pub(crate) initial_value_datum_roots: Arc<[DatumId]>,
    pub(crate) initial_value_list_roots: Arc<[ListId]>,
    // Sparse datum reads repeatedly resolve the same effective default while
    // the immutable catalogs remain unchanged. Cache owned answers (including
    // misses) without changing the borrowed public catalog API.
    pub(crate) effective_initial_value_cache:
        RefCell<HashMap<TypePath, HashMap<FieldName, Option<Value>>>>,
    pub(crate) effective_initial_value_cache_entries: Cell<usize>,
    // Type-scoped `initial()` reads are immutable after the runtime metadata is
    // installed. Cache their final value separately because a null catalog
    // entry can still require an inherited runtime initializer/prototype.
    pub(crate) initial_field_value_cache: HashMap<TypePath, HashMap<FieldName, Value>>,
    pub(crate) initial_field_value_cache_entries: usize,
    // Engine-created map cells whose identities are owned by the world keep
    // immutable effective defaults in `initial_values` and remain explicit GC
    // roots. Ordinary runtime datums use the same sparse field representation
    // but are rooted only by the normal DM object graph.
    pub(crate) compact_default_datums: HashSet<DatumId>,
    pub(crate) project_root: Option<Arc<PathBuf>>,
    pub(crate) dmm_measurements: Arc<BTreeMap<String, DmmMeasurement>>,
    pub(crate) parsed_dmm_cache: Arc<BTreeMap<String, ParsedDmm>>,
    pub(crate) random_state: u64,
    pub(crate) scheduler_tick: u64,
    pub(crate) scheduler_sequence: u64,
    pub(crate) scheduled_spawns: Vec<ScheduledSpawn>,
    // Native walking procedures run independently of DM continuations and do
    // not keep the startup scheduler drain open. A later call for the same
    // movable replaces the prior walk immediately.
    pub(crate) native_walks: HashMap<DatumId, NativeWalk>,
    // Due tasks are moved here while one scheduler dispatch is running. They
    // must remain visible to list GC until each continuation becomes active;
    // otherwise a collection in an earlier same-tick task can reclaim values
    // held only by a later due frame.
    pub(crate) scheduler_inflight: Vec<ScheduledSpawn>,
    // Wall-clock origin for the scheduler tick currently dispatching DM work.
    // Standalone execution leaves this disabled.
    pub(crate) scheduler_tick_started: Option<Instant>,
    // Optional production sampler spanning every scheduler slice of SSatoms.
    pub(crate) atoms_profile: Option<AtomsProfile>,
    // Independent exact `_tgm_load` subtree profiler.
    pub(crate) tgm_profile: Option<TgmProfile>,
    // Datums whose BYOND `Del()` hook is currently executing. This prevents a
    // reentrant `del(src)` from dispatching the same hook indefinitely.
    pub(crate) deleting_datums: HashSet<DatumId>,
    pub(crate) last_animation_target: Option<Value>,
    pub(crate) environment_overrides: BTreeMap<String, Option<Value>>,
    pub(crate) external_timers: BTreeMap<String, Instant>,
    pub(crate) iconforge_jobs: BTreeMap<String, (bool, String)>,
    pub(crate) iconforge_next_job: u64,
    pub(crate) iconforge_gags_configs: BTreeMap<String, PathBuf>,
    pub(crate) sql_jobs: BTreeMap<String, (bool, String)>,
    pub(crate) sql_next_job: u64,
    // Group slots by procedure so the hot cached-static read can borrow the
    // module's `&str` path. A flat `(String, slot)` key forced a fresh String
    // allocation on every invocation of procedures such as build_coordinate.
    pub(crate) procedure_static_locals: BTreeMap<String, BTreeMap<u16, Value>>,
    // Values returned across the public execution boundary remain valid until
    // the host explicitly replaces the execution state. DM has no API for the
    // host to announce that it dropped a returned list handle, so retain those
    // sparse roots while collecting the far more numerous frame temporaries.
    pub(crate) host_value_roots: Vec<Value>,
    pub(crate) next_list_collection: usize,
    /// Authoritative coordinate-to-cell identities for the mutable headless map.
    pub(crate) world_turfs: BTreeMap<(i32, i32, i32), DatumId>,
    // Ordered storage preserves BYOND traversal order while this dense
    // companion index makes the map loader's locate(x,y,z) hot path O(1).
    pub(crate) world_turf_lookup: Vec<Option<DatumId>>,
    pub(crate) world_turf_lookup_dimensions: (i32, i32, i32),
    // Successful NO_RUINS witnesses are safe to reuse only after revalidating
    // both the coordinate identity and its current flag value. We never cache
    // a successful rectangle, so newly-added flags cannot create a false pass.
    pub(crate) ruin_rejection_witnesses: BTreeMap<i32, BTreeMap<(i32, i32), DatumId>>,
    pub(crate) world_areas: BTreeMap<(i32, i32, i32), DatumId>,
    pub(crate) default_world_area: Option<DatumId>,
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self::from_heap(ValueHeap::default())
    }
}
impl ExecutionState {
    /// Creates an execution state with an empty value heap.
    #[must_use]
    pub fn new() -> Self {
        let state = Self::default();
        let _ = state.owner_thread.set(std::thread::current().id());
        state
    }

    /// Decodes a BYOND parameter string into its associative list representation.
    ///
    /// This is the host-side equivalent of `params2list()` and is used when the
    /// engine supplies `world.params` before `/world/New()` begins.
    pub fn decode_params_list(&mut self, params: &str) -> Result<Value, String> {
        builtins::params2list(&[Value::text(params)], self)
    }

    /// Creates execution state around an existing runtime heap.
    #[must_use]
    pub fn from_heap(heap: ValueHeap) -> Self {
        let mut state = Self {
            owner_thread: OnceLock::from(std::thread::current().id()),
            heap,
            associative_lists: HashSet::new(),
            reference_lists: HashSet::new(),
            savefiles: HashMap::new(),
            savefile_entries: HashMap::new(),
            client_sessions: BTreeMap::new(),
            interactive_local_clients: HashSet::new(),
            local_client_skin: None,
            local_client_outbound_events: BTreeMap::new(),
            local_client_mobs: BTreeMap::new(),
            local_client_commands: Vec::new(),
            local_client_command_sequence: 0,
            local_guest_sequence: 0,
            local_prompt_sequence: 0,
            pending_local_prompts: BTreeMap::new(),
            global_vars_proxy: None,
            datum_vars_proxies: HashMap::new(),
            datum_vars_by_datum: HashMap::new(),
            contents_owners: HashMap::new(),
            vis_contents_owners: HashMap::new(),
            vis_locs_owners: HashMap::new(),
            shared_fields: Arc::new(BTreeMap::new()),
            instance_initializers: Arc::new(BTreeMap::new()),
            instance_initializer_plans: HashMap::new(),
            initial_prototypes: BTreeMap::new(),
            initial_prototypes_initializing: HashSet::new(),
            instance_initializer_module: None,
            globals: GlobalStore::default(),
            initial_globals: BTreeMap::new(),
            type_paths: Arc::new(std::collections::BTreeSet::new()),
            type_parents: Arc::new(BTreeMap::new()),
            type_intervals: Arc::new(BTreeMap::new()),
            dynamic_receiver_targets: HashMap::new(),
            dynamic_callsite_targets: HashMap::new(),
            declared_field_slots: HashMap::new(),
            declared_field_quickening: DeclaredFieldQuickeningMetrics::default(),
            initial_values: Arc::new(BTreeMap::new()),
            initial_value_datum_roots: Arc::default(),
            initial_value_list_roots: Arc::default(),
            effective_initial_value_cache: RefCell::new(HashMap::new()),
            effective_initial_value_cache_entries: Cell::new(0),
            initial_field_value_cache: HashMap::new(),
            initial_field_value_cache_entries: 0,
            compact_default_datums: HashSet::new(),
            project_root: None,
            dmm_measurements: Arc::new(BTreeMap::new()),
            parsed_dmm_cache: Arc::new(BTreeMap::new()),
            random_state: 0,
            scheduler_tick: 0,
            scheduler_sequence: 0,
            scheduled_spawns: Vec::new(),
            native_walks: HashMap::new(),
            scheduler_inflight: Vec::new(),
            scheduler_tick_started: None,
            atoms_profile: None,
            tgm_profile: None,
            deleting_datums: HashSet::new(),
            last_animation_target: None,
            environment_overrides: BTreeMap::new(),
            external_timers: BTreeMap::new(),
            iconforge_jobs: BTreeMap::new(),
            iconforge_next_job: 0,
            iconforge_gags_configs: BTreeMap::new(),
            sql_jobs: BTreeMap::new(),
            sql_next_job: 0,
            procedure_static_locals: BTreeMap::new(),
            host_value_roots: Vec::new(),
            next_list_collection: 262_144,
            world_turfs: BTreeMap::new(),
            world_turf_lookup: Vec::new(),
            world_turf_lookup_dimensions: (0, 0, 0),
            ruin_rejection_witnesses: BTreeMap::new(),
            world_areas: BTreeMap::new(),
            default_world_area: None,
        };
        state.rebuild_world_geometry();
        state
    }

    pub(crate) fn assert_owner_thread(&self) {
        let owner = self
            .owner_thread
            .get_or_init(|| std::thread::current().id());
        assert_eq!(
            *owner,
            std::thread::current().id(),
            "live DM state mutation attempted off its owner thread"
        );
    }

    /// Allocates one inert map datum with sparse inherited scalar defaults.
    ///
    /// Effective scalar values remain available through the immutable initial
    /// value catalog, while runtime list/datum initializers still execute and
    /// receive fresh per-instance identities. Structural map fields such as
    /// coordinates and `loc` are supplied by the world allocator afterward.
    /// The datum is not inserted into `world.contents`; the bulk map handoff
    /// installs authoritative spatial membership once allocation completes.
    ///
    /// # Errors
    ///
    /// Returns a runtime initialization error when a per-instance initializer
    /// cannot be evaluated or stored.
    pub fn allocate_compact_map_datum(&mut self, type_path: TypePath) -> Result<DatumId, String> {
        let datum = self.heap.allocate_datum(type_path.clone());
        initialize_existing_datum(self, datum, type_path, true, true)?;
        Ok(datum)
    }

    /// Replaces the sparse-default identity set transferred with a runtime heap.
    pub fn set_compact_default_datums(&mut self, datums: HashSet<DatumId>) {
        self.compact_default_datums = datums;
    }

    /// Removes and returns the sparse-default identities for a host heap handoff.
    #[must_use]
    pub fn take_compact_default_datums(&mut self) -> HashSet<DatumId> {
        std::mem::take(&mut self.compact_default_datums)
    }

    /// Rebuilds the compact coordinate and contents-owner indexes from the
    /// datums currently present in the heap.
    ///
    /// Hosts that construct a deliberately minimal world (for example a
    /// client-lobby preflight) use this after installing coordinate fields.
    /// Normal map allocation already invokes the same operation during the
    /// runtime-image handoff.
    pub fn rebuild_world_geometry(&mut self) {
        self.world_turfs.clear();
        self.world_turf_lookup.clear();
        self.world_turf_lookup_dimensions = (0, 0, 0);
        self.world_areas.clear();
        self.contents_owners.clear();
        self.vis_contents_owners.clear();
        self.vis_locs_owners.clear();
        let contents = FieldName::parse("contents").expect("built-in contents field");
        let vis_contents = FieldName::parse("vis_contents").expect("built-in vis_contents field");
        let vis_locs = FieldName::parse("vis_locs").expect("built-in vis_locs field");
        for (id, datum) in self.heap.datums() {
            if let Ok(Value::List(list)) = datum.field(&contents) {
                self.contents_owners.insert(*list, id);
            }
            if let Ok(Value::List(list)) = datum.field(&vis_contents) {
                self.vis_contents_owners.insert(*list, id);
            }
            if let Ok(Value::List(list)) = datum.field(&vis_locs) {
                self.vis_locs_owners.insert(*list, id);
            }
        }
        let x = FieldName::parse("x").expect("built-in coordinate field");
        let y = FieldName::parse("y").expect("built-in coordinate field");
        let z = FieldName::parse("z").expect("built-in coordinate field");
        let loc = FieldName::parse("loc").expect("built-in loc field");
        for (id, datum) in self.heap.datums() {
            let path = datum.type_path().as_str();
            if path != "/turf" && !path.starts_with("/turf/") {
                continue;
            }
            let coordinate = [datum.field(&x), datum.field(&y), datum.field(&z)]
                .map(|value| value.ok().and_then(Value::as_number))
                .map(|value| value.filter(|value| value.is_finite() && value.fract() == 0.0));
            let [Some(x), Some(y), Some(z)] = coordinate else {
                continue;
            };
            #[allow(clippy::cast_possible_truncation)]
            let coordinate = (x as i32, y as i32, z as i32);
            self.world_turfs.insert(coordinate, id);
            if let Ok(Value::Datum(area)) = datum.field(&loc) {
                self.world_areas.insert(coordinate, *area);
            }
        }
        let world = self
            .heap
            .datums()
            .find(|(_, datum)| datum.type_path().as_str() == "/world")
            .map(|(id, _)| id);
        let extents = self.world_turfs.keys().fold(None, |extents, &(x, y, z)| {
            Some(
                extents.map_or((x, y, z), |(maxx, maxy, maxz): (i32, i32, i32)| {
                    (maxx.max(x), maxy.max(y), maxz.max(z))
                }),
            )
        });
        self.rebuild_world_turf_lookup();
        if let (Some(world), Some((maxx, maxy, maxz))) = (world, extents) {
            for (name, value) in [("maxx", maxx), ("maxy", maxy), ("maxz", maxz)] {
                let _ = self.heap.set_datum_field(
                    world,
                    FieldName::parse(name).expect("built-in world dimension field"),
                    Value::number(value as f32),
                );
            }
        }
    }

    pub(crate) fn world_dimension(&self, world: DatumId, name: &str) -> Result<i32, String> {
        let value = self
            .heap
            .datum_field(
                world,
                &FieldName::parse(name).expect("built-in world dimension field"),
            )
            .ok()
            .and_then(Value::as_number)
            .unwrap_or(1.0);
        if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > i32::MAX as f32 {
            return Err(format!(
                "world.{name} must be a positive integer, received {value}"
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        Ok(value as i32)
    }

    pub(crate) fn world_type_field(
        &self,
        world: DatumId,
        name: &str,
        fallback: &str,
    ) -> Result<TypePath, String> {
        let field = FieldName::parse(name).expect("built-in world type field");
        // Runtime images deliberately keep unchanged declared fields sparse.
        // World geometry creation must therefore observe the effective DM
        // initial value before applying the engine fallback, just like an
        // ordinary `world.area` or `world.turf` field read. Falling straight
        // back from a missing heap slot created dynamic z-levels as `/area`
        // and `/turf`, ignoring project declarations such as Monke's
        // `/area/space` and `/turf/open/space/basic`.
        match datum_field_or_initial(self, world, &field) {
            Ok(Value::TypePath(path)) => Ok(path),
            Ok(Value::ModifiedTypePath(path)) => Ok(path.base().clone()),
            Ok(Value::Null) | Err(_) => {
                TypePath::parse(fallback).map_err(|error| error.to_string())
            }
            Ok(value) => Err(format!(
                "world.{name} must be a type path, received {value}"
            )),
        }
    }

    pub(crate) fn ensure_contents(&mut self, datum: DatumId) -> Result<ListId, String> {
        let contents = FieldName::parse("contents").expect("built-in contents field");
        if let Ok(Value::List(list)) = self.heap.datum_field(datum, &contents) {
            self.contents_owners.insert(*list, datum);
            return Ok(*list);
        }
        let list = self.heap.allocate_list();
        self.heap
            .set_datum_field(datum, contents, Value::List(list))
            .map_err(|error| error.to_string())?;
        self.contents_owners.insert(list, datum);
        Ok(list)
    }

    pub(crate) fn contents_owner(&self, list: ListId) -> Option<DatumId> {
        self.contents_owners.get(&list).copied()
    }

    pub(crate) fn visibility_owner(&self, list: ListId) -> Option<(DatumId, bool)> {
        self.vis_contents_owners
            .get(&list)
            .copied()
            .map(|owner| (owner, true))
            .or_else(|| {
                self.vis_locs_owners
                    .get(&list)
                    .copied()
                    .map(|owner| (owner, false))
            })
    }

    pub(crate) fn is_visibility_list(&self, list: ListId) -> bool {
        self.visibility_owner(list).is_some()
    }

    pub(crate) fn visibility_list_accepts(&self, value: &Value) -> bool {
        let Value::Datum(datum) = value else {
            return matches!(value, Value::Null);
        };
        self.heap.datum(*datum).is_ok_and(|datum| {
            let path = datum.type_path().as_str();
            path == "/atom"
                || path.starts_with("/atom/")
                || path == "/obj"
                || path.starts_with("/obj/")
                || path == "/mob"
                || path.starts_with("/mob/")
                || path == "/turf"
                || path.starts_with("/turf/")
                || path == "/area"
                || path.starts_with("/area/")
        })
    }

    pub(crate) fn visibility_members(&self, list: ListId) -> Result<Vec<DatumId>, String> {
        Ok(self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::Datum(datum) => Some(*datum),
                _ => None,
            })
            .collect())
    }

    /// Applies one scalar `vis_contents` addition or removal without
    /// normalizing and diffing the complete relationship list.
    ///
    /// Returns `None` for ordinary lists and for `vis_locs`, whose direct
    /// mutation keeps using the general synchronization path. A handled
    /// `vis_contents` mutation returns whether its membership changed.
    pub(crate) fn mutate_vis_contents_scalar(
        &mut self,
        list: ListId,
        value: &Value,
        add: bool,
    ) -> Result<Option<bool>, String> {
        let Some((owner, true)) = self.visibility_owner(list) else {
            return Ok(None);
        };

        if add {
            if matches!(value, Value::Null) {
                return Ok(Some(false));
            }
            if !self.visibility_list_accepts(value) {
                return Err(format!(
                    "visibility lists can only contain atoms, received {value}"
                ));
            }
        }

        let Value::Datum(member) = value else {
            return Ok(Some(false));
        };
        let member_value = Value::Datum(*member);
        let contains = self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .contains(&member_value);
        if contains == add {
            return Ok(Some(false));
        }

        if add {
            self.heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .add(member_value);
        } else {
            self.heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_last(&member_value);
        }

        let reciprocal = self.ensure_visibility_list(*member, false)?;
        let owner_value = Value::Datum(owner);
        let reciprocal = self
            .heap
            .list_mut(reciprocal)
            .map_err(|error| error.to_string())?;
        if add {
            if !reciprocal.contains(&owner_value) {
                reciprocal.add(owner_value);
            }
        } else {
            while reciprocal.remove_last(&owner_value).is_some() {}
        }
        Ok(Some(true))
    }

    pub(crate) fn ensure_visibility_list(
        &mut self,
        datum: DatumId,
        vis_contents: bool,
    ) -> Result<ListId, String> {
        let name = if vis_contents {
            "vis_contents"
        } else {
            "vis_locs"
        };
        let field = FieldName::parse(name).expect("built-in visibility field");
        if let Ok(Value::List(list)) = self.heap.datum_field(datum, &field) {
            let list = *list;
            if vis_contents {
                self.vis_contents_owners.insert(list, datum);
            } else {
                self.vis_locs_owners.insert(list, datum);
            }
            return Ok(list);
        }
        let list = self.heap.allocate_list();
        self.heap
            .set_datum_field(datum, field, Value::List(list))
            .map_err(|error| error.to_string())?;
        if vis_contents {
            self.vis_contents_owners.insert(list, datum);
        } else {
            self.vis_locs_owners.insert(list, datum);
        }
        Ok(list)
    }

    pub(crate) fn synchronize_visibility_list(
        &mut self,
        list: ListId,
        before: &[DatumId],
    ) -> Result<(), String> {
        let Some((owner, is_vis_contents)) = self.visibility_owner(list) else {
            return Ok(());
        };
        let after = self.visibility_members(list)?;
        for removed in before {
            if after.contains(removed) {
                continue;
            }
            let reciprocal = self.ensure_visibility_list(*removed, !is_vis_contents)?;
            let reciprocal = self
                .heap
                .list_mut(reciprocal)
                .map_err(|error| error.to_string())?;
            while reciprocal.remove_last(&Value::Datum(owner)).is_some() {}
        }
        for added in after {
            if before.contains(&added) {
                continue;
            }
            let reciprocal = self.ensure_visibility_list(added, !is_vis_contents)?;
            let reciprocal = self
                .heap
                .list_mut(reciprocal)
                .map_err(|error| error.to_string())?;
            if !reciprocal.contains(&Value::Datum(owner)) {
                reciprocal.add(Value::Datum(owner));
            }
        }
        Ok(())
    }

    pub(crate) fn normalize_and_synchronize_visibility_list(
        &mut self,
        list: ListId,
        before: &[DatumId],
    ) -> Result<(), String> {
        if !self.is_visibility_list(list) {
            return Ok(());
        }
        let values = self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        let mut normalized = Vec::new();
        for value in values {
            if matches!(value, Value::Null) {
                continue;
            }
            if !self.visibility_list_accepts(&value) {
                return Err(format!(
                    "visibility lists can only contain atoms, received {value}"
                ));
            }
            if !normalized
                .iter()
                .any(|existing: &Value| existing.semantic_eq(&value))
            {
                normalized.push(value);
            }
        }
        let target = self
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?;
        target.resize(0).map_err(|error| error.to_string())?;
        for value in normalized {
            target.add(value);
        }
        self.synchronize_visibility_list(list, before)
    }

    pub(crate) fn default_area_for_world(&mut self, world: DatumId) -> Result<DatumId, String> {
        let path = self.world_type_field(world, "area", "/area")?;
        if let Some(area) = self.default_world_area
            && self
                .heap
                .datum(area)
                .is_ok_and(|datum| datum.type_path() == &path)
        {
            return Ok(area);
        }
        let existing = self
            .heap
            .datums()
            .find_map(|(id, datum)| (datum.type_path() == &path).then_some(id));
        let area = match existing {
            Some(area) => area,
            None => allocate_initialized_datum(self, path)?,
        };
        self.ensure_contents(area)?;
        let world_contents = self.ensure_contents(world)?;
        let contents = self
            .heap
            .list_mut(world_contents)
            .map_err(|error| error.to_string())?;
        if !contents.contains(&Value::Datum(area)) {
            contents.add(Value::Datum(area));
        }
        self.default_world_area = Some(area);
        Ok(area)
    }

    pub(crate) fn remove_world_cell(
        &mut self,
        world: DatumId,
        coordinate: (i32, i32, i32),
    ) -> Result<(), String> {
        let Some(turf) = self.world_turfs.remove(&coordinate) else {
            self.world_areas.remove(&coordinate);
            return Ok(());
        };
        if let Some(area) = self.world_areas.remove(&coordinate) {
            let contents = self.ensure_contents(area)?;
            self.heap
                .list_mut(contents)
                .map_err(|error| error.to_string())?
                .remove_first(&Value::Datum(turf));
        }
        let contents = self.ensure_contents(world)?;
        self.heap
            .list_mut(contents)
            .map_err(|error| error.to_string())?
            .remove_first(&Value::Datum(turf));
        self.heap
            .destroy_datum(turf)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub(crate) fn resize_world_geometry(
        &mut self,
        world: DatumId,
        dimensions: (i32, i32, i32),
    ) -> Result<(), String> {
        let (maxx, maxy, maxz) = dimensions;
        if maxx < 1 || maxy < 1 || maxz < 1 {
            return Err("world dimensions must be positive integers".to_owned());
        }
        let removed = self
            .world_turfs
            .keys()
            .copied()
            .filter(|(x, y, z)| *x > maxx || *y > maxy || *z > maxz)
            .collect::<Vec<_>>();
        for coordinate in removed {
            self.remove_world_cell(world, coordinate)?;
        }

        let area = self.default_area_for_world(world)?;
        let turf_path = self.world_type_field(world, "turf", "/turf")?;
        let area_contents = self.ensure_contents(area)?;
        // Constant initializer actions are deliberately omitted for fresh
        // compact engine turfs: their values remain in the shared initial
        // catalog. Only runtime initializer programs can make allocation
        // observable and require the general per-cell path.
        let turf_has_runtime_initializers = instance_initializer_plan(self, &turf_path)
            .iter()
            .any(|initializer| matches!(initializer, InstanceInitializer::Program { .. }));
        let mut bulk_area_members = Vec::new();
        let bulk_world_contents = (!turf_has_runtime_initializers)
            .then(|| {
                self.global(&FieldName::parse("world").ok()?)
                    .and_then(|value| (value == &Value::Datum(world)).then_some(()))?;
                self.heap
                    .datum_field(world, &FieldName::parse("contents").ok()?)
                    .ok()
                    .and_then(|value| match value {
                        Value::List(list) => Some(*list),
                        _ => None,
                    })
            })
            .flatten();
        let mut bulk_world_members = Vec::new();
        let coordinate_fields = ["x", "y", "z"]
            .map(|name| FieldName::parse(name).expect("built-in coordinate field is valid"));
        let loc = FieldName::parse("loc").expect("built-in loc field is valid");
        for z in 1..=maxz {
            for y in 1..=maxy {
                for x in 1..=maxx {
                    let coordinate = (x, y, z);
                    if self.world_turfs.contains_key(&coordinate) {
                        continue;
                    }
                    let turf = if turf_has_runtime_initializers {
                        allocate_initialized_datum(self, turf_path.clone())?
                    } else {
                        self.heap.allocate_datum(turf_path.clone())
                    };
                    for (field, value) in coordinate_fields.iter().zip([x, y, z]) {
                        self.heap
                            .set_datum_field(turf, field.clone(), Value::number(value as f32))
                            .map_err(|error| error.to_string())?;
                    }
                    self.heap
                        .set_datum_field(turf, loc.clone(), Value::Datum(area))
                        .map_err(|error| error.to_string())?;
                    if turf_has_runtime_initializers {
                        self.heap
                            .list_mut(area_contents)
                            .map_err(|error| error.to_string())?
                            .add(Value::Datum(turf));
                    } else {
                        bulk_area_members.push(Value::Datum(turf));
                        if bulk_world_contents.is_some() {
                            bulk_world_members.push(Value::Datum(turf));
                        }
                    }
                    self.world_turfs.insert(coordinate, turf);
                    self.world_areas.insert(coordinate, area);
                }
            }
        }
        if !bulk_area_members.is_empty() {
            self.heap
                .list_mut(area_contents)
                .map_err(|error| error.to_string())?
                .extend_positional(bulk_area_members);
        }
        if let Some(world_contents) = bulk_world_contents
            && !bulk_world_members.is_empty()
        {
            self.heap
                .list_mut(world_contents)
                .map_err(|error| error.to_string())?
                .extend_positional(bulk_world_members);
        }
        for (name, value) in [("maxx", maxx), ("maxy", maxy), ("maxz", maxz)] {
            self.heap
                .set_datum_field(
                    world,
                    FieldName::parse(name).expect("built-in world dimension field"),
                    Value::number(value as f32),
                )
                .map_err(|error| error.to_string())?;
        }
        self.rebuild_world_turf_lookup();
        Ok(())
    }

    pub(crate) fn turf_at(&self, x: i32, y: i32, z: i32) -> Option<DatumId> {
        let coordinate = (x, y, z);
        let (maxx, maxy, maxz) = self.world_turf_lookup_dimensions;
        if x >= 1 && y >= 1 && z >= 1 && x <= maxx && y <= maxy && z <= maxz {
            let index = ((z - 1) as usize * maxy as usize + (y - 1) as usize) * maxx as usize
                + (x - 1) as usize;
            if let Some(turf) = self.world_turf_lookup.get(index).copied().flatten() {
                return Some(turf);
            }
        }
        self.world_turfs.get(&coordinate).copied()
    }

    pub(crate) fn rebuild_world_turf_lookup(&mut self) {
        let dimensions = self
            .world_turfs
            .keys()
            .fold((0, 0, 0), |(maxx, maxy, maxz), &(x, y, z)| {
                (maxx.max(x), maxy.max(y), maxz.max(z))
            });
        self.world_turf_lookup_dimensions = dimensions;
        let (maxx, maxy, maxz) = dimensions;
        let Some(length) = usize::try_from(maxx)
            .ok()
            .and_then(|x| usize::try_from(maxy).ok().and_then(|y| x.checked_mul(y)))
            .and_then(|xy| usize::try_from(maxz).ok().and_then(|z| xy.checked_mul(z)))
        else {
            self.world_turf_lookup.clear();
            return;
        };
        self.world_turf_lookup.clear();
        self.world_turf_lookup.resize(length, None);
        for (&(x, y, z), &turf) in &self.world_turfs {
            if x < 1 || y < 1 || z < 1 {
                continue;
            }
            let index = ((z - 1) as usize * maxy as usize + (y - 1) as usize) * maxx as usize
                + (x - 1) as usize;
            if let Some(slot) = self.world_turf_lookup.get_mut(index) {
                *slot = Some(turf);
            }
        }
    }

    pub(crate) fn note_turf_area(&mut self, turf: DatumId, area: DatumId) {
        let coordinate = ["x", "y", "z"]
            .map(|name| FieldName::parse(name).expect("built-in coordinate field"))
            .map(|field| {
                self.heap
                    .datum_field(turf, &field)
                    .ok()
                    .and_then(Value::as_number)
            });
        let [Some(x), Some(y), Some(z)] = coordinate else {
            return;
        };
        if [x, y, z]
            .into_iter()
            .any(|value| !value.is_finite() || value.fract() != 0.0)
        {
            return;
        }
        #[allow(clippy::cast_possible_truncation)]
        self.world_areas
            .insert((x as i32, y as i32, z as i32), area);
    }

    /// Returns ownership of the runtime heap after execution.
    #[must_use]
    pub fn into_heap(self) -> ValueHeap {
        self.heap
    }

    /// Transfers persistent procedure-static local storage out of this state.
    ///
    /// Runtime image/lifecycle phase boundaries temporarily separate the heap
    /// from VM execution metadata. BYOND procedure statics outlive those
    /// boundaries and must travel with the materialized world rather than be
    /// recreated as null on the next phase.
    pub fn take_procedure_static_locals(&mut self) -> BTreeMap<(String, u16), Value> {
        std::mem::take(&mut self.procedure_static_locals)
            .into_iter()
            .flat_map(|(path, slots)| {
                slots
                    .into_iter()
                    .map(move |(slot, value)| ((path.clone(), slot), value))
            })
            .collect()
    }

    /// Restores persistent procedure-static locals captured from an earlier
    /// execution phase.
    pub fn set_procedure_static_locals(&mut self, locals: BTreeMap<(String, u16), Value>) {
        self.procedure_static_locals.clear();
        for ((path, slot), value) in locals {
            self.procedure_static_locals
                .entry(path)
                .or_default()
                .insert(slot, value);
        }
    }

    /// Returns the shared value heap.
    #[must_use]
    pub const fn heap(&self) -> &ValueHeap {
        &self.heap
    }

    /// Returns the shared mutable value heap.
    #[must_use]
    pub fn heap_mut(&mut self) -> &mut ValueHeap {
        self.assert_owner_thread();
        &mut self.heap
    }

    /// Allocates a local `/client` and atomically installs its parsed DMF skin.
    ///
    /// # Errors
    ///
    /// Returns the parser diagnostics without allocating a client when the
    /// supplied DMF contains any errors.
    ///
    /// # Panics
    ///
    /// Panics only if the engine's built-in `/client` type path becomes invalid.
    pub fn open_local_client(&mut self, dmf_source: &str) -> Result<DatumId, LocalClientError> {
        let document = parse_dmf(dmf_source);
        let diagnostics: Vec<_> = document
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
            .cloned()
            .collect();
        if !diagnostics.is_empty() {
            return Err(LocalClientError { diagnostics });
        }
        let client = self
            .heap
            .allocate_datum(TypePath::parse("/client").expect("the engine client path is valid"));
        self.install_client_session(client, ControlTree::from_document(&document));
        Ok(client)
    }

    /// Installs a skin-backed UI session for a connected client datum.
    pub fn install_client_session(&mut self, client: DatumId, tree: ControlTree) {
        self.client_sessions
            .insert(client, ClientSession::new(tree));
    }

    /// Sets the parsed skin cloned into subsequently connected local clients.
    pub fn set_local_client_skin(&mut self, tree: ControlTree) {
        self.local_client_skin = Some(tree);
    }

    /// Drains authoritative UI operations in exact DM execution order.
    #[must_use]
    pub fn take_local_client_outbound_events(
        &mut self,
        client: DatumId,
    ) -> Vec<LocalClientUiEvent> {
        self.local_client_outbound_events
            .remove(&client)
            .unwrap_or_default()
    }

    pub(crate) fn emit_local_client_ui_event(
        &mut self,
        client: DatumId,
        event: LocalClientUiEvent,
    ) {
        if self.client_sessions.contains_key(&client) {
            self.local_client_outbound_events
                .entry(client)
                .or_default()
                .push(event);
        }
    }

    /// Returns the number of DM continuations waiting for native prompt input.
    #[must_use]
    pub fn pending_local_prompt_count(&self) -> usize {
        self.pending_local_prompts.len()
    }

    /// Supplies one typed native prompt answer and schedules its suspended DM
    /// continuation at the current scheduler boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown prompt, the wrong client, cancellation
    /// of a required prompt, an invalid number, or an out-of-range choice.
    pub fn submit_local_prompt_response(
        &mut self,
        client: DatumId,
        id: u64,
        response: LocalClientPromptResponse,
    ) -> Result<(), String> {
        let prompt = self
            .pending_local_prompts
            .remove(&id)
            .ok_or_else(|| format!("unknown local prompt {id}"))?;
        if prompt.client != client {
            self.pending_local_prompts.insert(id, prompt);
            return Err(format!("local prompt {id} belongs to another client"));
        }
        if matches!(response, LocalClientPromptResponse::Null)
            && matches!(&prompt.continuation, PendingPromptContinuation::Verb(_))
            && prompt.can_cancel
        {
            return Ok(());
        }
        let value = match response {
            LocalClientPromptResponse::Null if prompt.can_cancel => Value::Null,
            LocalClientPromptResponse::Null => {
                self.pending_local_prompts.insert(id, prompt);
                return Err(format!("local prompt {id} cannot be cancelled"));
            }
            LocalClientPromptResponse::Text(value)
                if matches!(
                    prompt.kind,
                    LocalClientPromptKind::Text
                        | LocalClientPromptKind::Message
                        | LocalClientPromptKind::Color
                        | LocalClientPromptKind::File
                ) =>
            {
                if prompt.kind == LocalClientPromptKind::File {
                    Value::File(value.into())
                } else {
                    Value::text(value)
                }
            }
            LocalClientPromptResponse::Text(value)
                if prompt.kind == LocalClientPromptKind::Number =>
            {
                Value::number(value.parse::<f32>().map_err(|_| {
                    self.pending_local_prompts.insert(id, prompt.clone());
                    format!("local prompt {id} requires a number")
                })?)
            }
            LocalClientPromptResponse::Number(value)
                if prompt.kind == LocalClientPromptKind::Number && value.is_finite() =>
            {
                Value::number(value)
            }
            LocalClientPromptResponse::Choice(index)
                if matches!(
                    prompt.kind,
                    LocalClientPromptKind::List | LocalClientPromptKind::Alert
                ) =>
            {
                prompt.choices.get(index).cloned().ok_or_else(|| {
                    self.pending_local_prompts.insert(id, prompt.clone());
                    format!("local prompt {id} choice {index} is out of range")
                })?
            }
            _ => {
                self.pending_local_prompts.insert(id, prompt);
                return Err(format!(
                    "local prompt {id} received an incompatible response"
                ));
            }
        };
        match prompt.continuation {
            PendingPromptContinuation::Frames(mut frames) => {
                let frame = frames
                    .last_mut()
                    .ok_or_else(|| format!("local prompt {id} has no continuation"))?;
                frame.stack.push(value);
                schedule_frames(self, frames, 0.0);
            }
            PendingPromptContinuation::Verb(mut invocation) => {
                let parameter = invocation.parameter;
                let value = if invocation.parameter_types[parameter] == VerbParameterType::File {
                    match value {
                        Value::Text(path) => Value::File(path),
                        value => value,
                    }
                } else {
                    value
                };
                let Some(local) = invocation.frame.locals.get_mut(parameter) else {
                    return Err(format!("local prompt {id} verb parameter is invalid"));
                };
                *local = value.clone();
                if parameter >= invocation.frame.arguments.len() {
                    invocation
                        .frame
                        .arguments
                        .resize(parameter + 1, Value::Null);
                }
                invocation.frame.arguments[parameter] = value;
                invocation.frame.supplied_parameters[parameter] = true;
                queue_next_verb_prompt(self, client, invocation)?;
            }
        }
        Ok(())
    }

    /// Returns the UI session associated with a connected client datum.
    #[must_use]
    pub fn client_session(&self, client: DatumId) -> Option<&ClientSession> {
        self.client_sessions.get(&client)
    }

    /// Returns the mutable UI session associated with a connected client datum.
    pub fn client_session_mut(&mut self, client: DatumId) -> Option<&mut ClientSession> {
        self.client_sessions.get_mut(&client)
    }

    /// Enables or disables modal prompt suspension for a window-attached
    /// client. Skin-only preflight clients remain non-interactive so startup
    /// probes cannot deadlock waiting for a UI response that has no consumer.
    pub fn set_local_client_interactive(
        &mut self,
        client: DatumId,
        interactive: bool,
    ) -> Result<(), String> {
        if !self.client_sessions.contains_key(&client) {
            return Err("local client has no installed UI session".to_owned());
        }
        if interactive {
            self.interactive_local_clients.insert(client);
        } else {
            self.interactive_local_clients.remove(&client);
        }
        Ok(())
    }

    /// Drains local UI events emitted by one connected client.
    #[must_use]
    pub fn take_client_events(&mut self, client: DatumId) -> Vec<UiEvent> {
        self.client_sessions
            .get_mut(&client)
            .map_or_else(Vec::new, ClientSession::take_events)
    }

    /// Binds an existing local client to an existing mob datum.
    ///
    /// # Errors
    ///
    /// Returns an error for stale identities or non-client/non-mob runtime types.
    pub fn attach_local_client(
        &mut self,
        client: DatumId,
        mob: DatumId,
    ) -> Result<LocalClientState, String> {
        let client_path = self
            .heap
            .datum(client)
            .map_err(|error| error.to_string())?
            .type_path()
            .as_str();
        if client_path != "/client" && !client_path.starts_with("/client/") {
            return Err("local controller identity is not a /client".to_owned());
        }
        let mob_path = self
            .heap
            .datum(mob)
            .map_err(|error| error.to_string())?
            .type_path()
            .as_str();
        if mob_path != "/mob" && !mob_path.starts_with("/mob/") {
            return Err("local controlled identity is not a /mob".to_owned());
        }
        assign_datum_field(
            self,
            client,
            FieldName::parse("mob").unwrap(),
            Value::Datum(mob),
        )?;
        assign_datum_field(
            self,
            mob,
            FieldName::parse("client").unwrap(),
            Value::Datum(client),
        )?;
        if self.local_client_coordinates(mob).is_err() {
            let turf = self
                .world_turfs
                .first_key_value()
                .map(|(_, turf)| *turf)
                .ok_or_else(|| "local client cannot attach to an empty world".to_owned())?;
            assign_datum_field(
                self,
                mob,
                FieldName::parse("loc").unwrap(),
                Value::Datum(turf),
            )?;
        }
        self.local_client_mobs.insert(client, mob);
        self.local_client_state(client)
    }

    /// Allocates a local client and mob and places the mob on the first indexed turf.
    ///
    /// # Errors
    ///
    /// Returns an error when the authoritative world has no materialized turf.
    pub fn create_attached_local_client(&mut self) -> Result<LocalClientState, String> {
        let turf = self.world_turfs.values().next().copied().ok_or_else(|| {
            "cannot attach a local client before the map is materialized".to_owned()
        })?;
        let client = allocate_initialized_datum(
            self,
            TypePath::parse("/client").expect("engine client path is valid"),
        )?;
        let mob_type = self.connection_mob_type();
        let mob = allocate_initialized_datum(self, mob_type)?;
        assign_datum_field(
            self,
            mob,
            FieldName::parse("loc").unwrap(),
            Value::Datum(turf),
        )?;
        self.attach_local_client(client, mob)
    }

    pub(crate) fn create_pending_local_client(&mut self) -> Result<LocalClientState, String> {
        let turf = self.world_turfs.values().next().copied().ok_or_else(|| {
            "cannot attach a local client before the map is materialized".to_owned()
        })?;
        let client = allocate_initialized_datum(
            self,
            TypePath::parse("/client").expect("engine client path is valid"),
        )?;
        let mob = allocate_initialized_datum(self, self.connection_mob_type())?;
        assign_datum_field(
            self,
            mob,
            FieldName::parse("loc").unwrap(),
            Value::Datum(turf),
        )?;
        self.local_client_mobs.insert(client, mob);
        self.local_client_state(client)
    }

    pub(crate) fn connection_mob_type(&self) -> TypePath {
        let fallback = TypePath::parse("/mob").expect("engine mob path is valid");
        let mob_field = FieldName::parse("mob").expect("engine world mob field is valid");
        self.heap
            .datums()
            .find(|(_, datum)| {
                let path = datum.type_path().as_str();
                path == "/world" || path.starts_with("/world/")
            })
            .and_then(|(world, _)| datum_field_or_initial(self, world, &mob_field).ok())
            .and_then(|value| match value {
                Value::TypePath(path) if builtins::is_subtype(self, &path, &fallback) => Some(path),
                Value::ModifiedTypePath(path)
                    if builtins::is_subtype(self, path.base(), &fallback) =>
                {
                    Some(path.base().clone())
                }
                _ => None,
            })
            .unwrap_or(fallback)
    }

    /// Creates a deterministic loopback guest and queues its project-defined
    /// `/client/New()` hook at the current scheduler boundary.
    ///
    /// The client session and client/mob relationship are installed before the
    /// frame becomes runnable, so `New()` observes the same fully connected
    /// identity that later UI builtins use. A sleeping hook remains an ordinary
    /// scheduled continuation; runtime failures are returned by
    /// [`advance_scheduler`].
    ///
    /// # Errors
    ///
    /// Returns an error when the world has no turf, the client cannot be bound,
    /// or the runtime client type has no effective `New` implementation.
    pub fn connect_local_guest(&mut self, module: &Module) -> Result<LocalClientState, String> {
        if self.local_client_skin.is_none()
            && let Some(root) = self.project_root.as_deref()
        {
            let skin_path = root.join("interface").join("skin.dmf");
            if skin_path.is_file() {
                let source = std::fs::read_to_string(&skin_path).map_err(|error| {
                    format!(
                        "failed to read local client skin {}: {error}",
                        skin_path.display()
                    )
                })?;
                let document = parse_dmf(&source);
                if let Some(diagnostic) = document
                    .diagnostics
                    .iter()
                    .find(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
                {
                    return Err(format!(
                        "local client skin {} is invalid: {}",
                        skin_path.display(),
                        diagnostic.message
                    ));
                }
                self.local_client_skin = Some(ControlTree::from_document(&document));
            }
        }
        let attached = self.create_pending_local_client()?;
        self.populate_local_verb_inventory(module, attached.client)?;
        self.populate_local_verb_inventory(module, attached.mob)?;
        let sequence = self.local_guest_sequence.saturating_add(1);
        self.local_guest_sequence = sequence;
        let key = format!("Guest-{sequence}");
        for (name, value) in [
            ("key", Value::text(key.as_str())),
            ("ckey", Value::text(key.to_ascii_lowercase())),
            ("address", Value::text("127.0.0.1")),
            (
                "computer_id",
                Value::text(format!("dream64-local-{sequence}")),
            ),
            ("connection", Value::text("seeker")),
            ("byond_version", Value::number(516.0)),
            ("byond_build", Value::number(1680.0)),
        ] {
            let field = FieldName::parse(name).expect("guest identity field is valid");
            if datum_field_or_initial(self, attached.client, &field).is_ok() {
                assign_datum_field(self, attached.client, field, value)?;
            }
        }
        self.install_client_session(
            attached.client,
            self.local_client_skin.clone().unwrap_or_default(),
        );

        let receiver = Value::Datum(attached.client);
        let (procedure, context) = dynamic_call_target_named(
            module,
            self,
            &receiver,
            "New",
            &ExecutionContext::new(receiver.clone(), receiver.clone()),
            false,
        )?;
        let program = module.resolve_procedure(procedure)?;
        let frame = make_frame(procedure, program, &[], &context);
        schedule_frames(self, vec![frame], 0.0);
        Ok(attached)
    }

    /// Queues cardinal movement for deterministic application at a scheduler boundary.
    ///
    /// # Errors
    ///
    /// Returns an error when the client is not attached or the controlled mob is stale.
    pub fn queue_local_movement(
        &mut self,
        client: DatumId,
        direction: LocalMovementDirection,
    ) -> Result<(), String> {
        let mob = self
            .local_client_mobs
            .get(&client)
            .copied()
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        self.heap.datum(mob).map_err(|error| error.to_string())?;
        let sequence = self.local_client_command_sequence;
        self.local_client_command_sequence = sequence.saturating_add(1);
        self.local_client_commands
            .push((sequence, client, direction));
        Ok(())
    }

    /// Queues a browser `byond://` request through the attached client's
    /// effective `Topic()` implementation.
    ///
    /// # Errors
    ///
    /// Returns an error when the client is detached, the parameter string is
    /// invalid, or no effective client `Topic` procedure can be resolved.
    pub fn queue_local_browser_topic(
        &mut self,
        module: &Module,
        client: DatumId,
        topic: &str,
    ) -> Result<(), String> {
        let mob = *self
            .local_client_mobs
            .get(&client)
            .ok_or_else(|| "local client is not attached".to_owned())?;
        self.heap.datum(client).map_err(|error| error.to_string())?;
        self.heap.datum(mob).map_err(|error| error.to_string())?;
        let query = topic
            .strip_prefix("byond://")
            .unwrap_or(topic)
            .strip_prefix('?')
            .unwrap_or_else(|| topic.strip_prefix("byond://").unwrap_or(topic));
        let href_list = self.decode_params_list(query)?;
        let hsrc = match &href_list {
            Value::List(list) => self
                .heap
                .list(*list)
                .ok()
                .and_then(|values| values.get_key(&Value::text("src")).ok())
                .and_then(|value| match value {
                    Value::Text(reference) => match parse_heap_reference(&reference) {
                        Some(HeapReference::Datum(index)) => {
                            self.heap.datum_id_at_index(index).map(Value::Datum)
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .unwrap_or(Value::Null),
            _ => Value::Null,
        };
        let receiver = Value::Datum(client);
        let usr = Value::Datum(mob);
        let (procedure, context) = dynamic_call_target_named(
            module,
            self,
            &receiver,
            "Topic",
            // BYOND dispatches /client/Topic() with src set to the client and
            // usr set to that client's mob. SS13 security middleware rejects
            // browser messages when this relationship is not preserved.
            &ExecutionContext::new(receiver.clone(), usr),
            false,
        )?;
        let program = module.resolve_procedure(procedure)?;
        let arguments = [Value::text(topic), href_list, hsrc, Value::number(0.0)];
        schedule_frames(
            self,
            vec![make_frame(procedure, program, &arguments, &context)],
            0.0,
        );
        Ok(())
    }

    /// Resolves and queues one BYOND command against the attached client's
    /// verb inventory. Client verbs take precedence over mob verbs, matching
    /// the command surface exposed by a connected BYOND client.
    ///
    /// # Errors
    ///
    /// Returns an error when the client is detached, the command is malformed,
    /// no matching verb exists, or its supplied argument count is invalid.
    pub fn queue_local_client_command(
        &mut self,
        module: &Module,
        client: DatumId,
        command: &str,
    ) -> Result<(), String> {
        let mob = *self
            .local_client_mobs
            .get(&client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        self.heap.datum(client).map_err(|error| error.to_string())?;
        self.heap.datum(mob).map_err(|error| error.to_string())?;

        let (command_name, raw_arguments) = split_client_command(command)?;
        let normalized_command = normalize_client_command_name(command_name);
        let client_receiver = Value::Datum(client);
        let mob_receiver = Value::Datum(mob);
        let caller = ExecutionContext::new(client_receiver.clone(), mob_receiver.clone());
        let mut resolved = None;
        for receiver in [&client_receiver, &mob_receiver] {
            let Value::Datum(datum) = receiver else {
                unreachable!("local command receivers are datums")
            };
            for verb_path in self.local_verb_inventory(*datum)? {
                let path = verb_path.as_str();
                let Some((_, selector)) = path.rsplit_once("/verb/") else {
                    continue;
                };
                let selector = selector.split('@').next().unwrap_or(selector);
                let Some(procedure) = module
                    .effective_procedure_id(path)
                    .or_else(|| module.procedure_id(path))
                else {
                    continue;
                };
                let Some(program) = module.procedure(procedure) else {
                    continue;
                };
                let verb_command_name = program.verb_name.as_deref().unwrap_or(selector);
                if normalize_client_command_name(verb_command_name) != normalized_command {
                    continue;
                }
                let explicit_selector = format!("verb/{selector}");
                if let Ok(target) = dynamic_call_target_named(
                    module,
                    self,
                    receiver,
                    &explicit_selector,
                    &caller,
                    false,
                ) {
                    resolved = Some(target);
                    break;
                }
            }
            if resolved.is_some() {
                break;
            }
        }
        let (procedure, context) =
            resolved.ok_or_else(|| format!("unknown client command {command_name:?}"))?;
        let program = module.resolve_procedure(procedure)?;
        let arguments = parse_client_command_arguments(raw_arguments)?;
        if arguments.len() > program.parameter_count {
            return Err(format!(
                "client command {command_name:?} accepts at most {} argument(s), received {}",
                program.parameter_count,
                arguments.len()
            ));
        }
        let mut values = vec![Value::Null; program.parameter_count];
        let mut supplied = vec![false; program.parameter_count];
        for (index, argument) in arguments.into_iter().enumerate() {
            match program.verb_parameter_types[index] {
                VerbParameterType::Text | VerbParameterType::Message | VerbParameterType::Color => {
                    values[index] = Value::text(argument);
                    supplied[index] = true;
                }
                VerbParameterType::Number => {
                    values[index] = Value::number(argument.parse::<f32>().map_err(|_| {
                        format!(
                            "invalid number argument {argument:?} for client command {command_name:?}",
                        )
                    })?);
                    supplied[index] = true;
                }
                VerbParameterType::File => {
                    values[index] = Value::File(argument.into());
                    supplied[index] = true;
                }
                VerbParameterType::Atom(_)
                | VerbParameterType::Anything
                | VerbParameterType::Unsupported => {}
            }
        }
        let mut frame = make_frame(procedure, program, &values, &context);
        frame.supplied_parameters = supplied.into();
        if frame.supplied_parameters.iter().all(|supplied| *supplied) {
            schedule_frames(self, vec![frame], 0.0);
        } else {
            queue_next_verb_prompt(
                self,
                client,
                PendingVerbInvocation {
                    frame,
                    parameter_types: program.verb_parameter_types.clone(),
                    parameter_names: program.parameter_names.clone(),
                    verb_name: program
                        .verb_name
                        .clone()
                        .unwrap_or_else(|| command_name.to_owned()),
                    parameter: 0,
                },
            )?;
        }
        Ok(())
    }

    pub(crate) fn populate_local_verb_inventory(
        &mut self,
        module: &Module,
        datum: DatumId,
    ) -> Result<(), String> {
        let runtime_type = self
            .heap
            .datum(datum)
            .map_err(|error| error.to_string())?
            .type_path()
            .clone();
        let mut defaults = Vec::new();
        for path in module.procedure_paths() {
            let canonical = path.split_once('@').map_or(path, |(path, _)| path);
            let Some((owner, _)) = canonical.rsplit_once("/verb/") else {
                continue;
            };
            let Ok(owner) = TypePath::parse(owner) else {
                continue;
            };
            if builtins::is_subtype(self, &runtime_type, &owner)
                && let Ok(verb) = TypePath::parse(canonical)
                && !defaults.contains(&verb)
            {
                defaults.push(verb);
            }
        }
        let verbs_field = FieldName::parse("verbs").expect("engine verbs field is valid");
        let list = match self.heap.datum_field(datum, &verbs_field) {
            Ok(Value::List(list)) => *list,
            _ => {
                let list = self.heap.allocate_list();
                self.heap
                    .set_datum_field(datum, verbs_field, Value::List(list))
                    .map_err(|error| error.to_string())?;
                list
            }
        };
        let existing = self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        let list = self
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?;
        for verb in defaults {
            let value = Value::TypePath(verb);
            if !existing.contains(&value) {
                list.add(value);
            }
        }
        Ok(())
    }

    pub(crate) fn local_verb_inventory(&self, datum: DatumId) -> Result<Vec<TypePath>, String> {
        let verbs = datum_field_or_initial(
            self,
            datum,
            &FieldName::parse("verbs").expect("engine verbs field is valid"),
        )
        .map_err(|error| error.to_string())?;
        let Value::List(verbs) = verbs else {
            return Ok(Vec::new());
        };
        Ok(self
            .heap
            .list(verbs)
            .map_err(|error| error.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::TypePath(path) => Some(path.clone()),
                Value::ModifiedTypePath(path) => Some(path.base().clone()),
                _ => None,
            })
            .collect())
    }

    /// Applies every queued local command in stable enqueue order.
    ///
    /// This is the host's scheduler-boundary commit point. No movement mutates
    /// the live world before this method is called.
    ///
    /// # Errors
    ///
    /// Returns an error if an attached datum or its authoritative turf is stale.
    pub fn apply_local_client_commands(&mut self) -> Result<Vec<LocalClientState>, String> {
        let mut commands = std::mem::take(&mut self.local_client_commands);
        commands.sort_by_key(|(sequence, _, _)| *sequence);
        let mut committed = Vec::with_capacity(commands.len());
        for (_, client, direction) in commands {
            let mob = *self
                .local_client_mobs
                .get(&client)
                .ok_or_else(|| "local client detached before movement commit".to_owned())?;
            let current = self.local_client_coordinates(mob)?;
            let (dx, dy) = match direction {
                LocalMovementDirection::North => (0, 1),
                LocalMovementDirection::South => (0, -1),
                LocalMovementDirection::East => (1, 0),
                LocalMovementDirection::West => (-1, 0),
            };
            if let Some(destination) = self.turf_at(current.0 + dx, current.1 + dy, current.2) {
                assign_datum_field(
                    self,
                    mob,
                    FieldName::parse("loc").unwrap(),
                    Value::Datum(destination),
                )?;
            }
            committed.push(self.local_client_state(client)?);
        }
        Ok(committed)
    }

    /// Returns the authoritative location for an attached local client.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown client or a mob outside a turf.
    pub fn local_client_state(&self, client: DatumId) -> Result<LocalClientState, String> {
        let mob = *self
            .local_client_mobs
            .get(&client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        let (x, y, z) = self.local_client_coordinates(mob)?;
        Ok(LocalClientState {
            client,
            mob,
            x,
            y,
            z,
        })
    }

    /// Returns the turf coordinates observed by a client camera. BYOND uses
    /// `client.eye` for map projection and falls back to the controlled mob
    /// when no explicit eye is installed.
    pub fn local_client_view_coordinates(
        &self,
        client: DatumId,
    ) -> Result<(i32, i32, i32), String> {
        let mob = *self
            .local_client_mobs
            .get(&client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        let eye = datum_field_or_initial(
            self,
            client,
            &FieldName::parse("eye").expect("client eye field"),
        )
        .map_err(|error| error.to_string())?;
        if let Value::Datum(eye) = eye {
            if let Some(coordinate) = self
                .world_turfs
                .iter()
                .find_map(|(coordinate, turf)| (*turf == eye).then_some(*coordinate))
            {
                return Ok(coordinate);
            }
            if let Ok(coordinate) = self.local_client_coordinates(eye) {
                return Ok(coordinate);
            }
        }
        self.local_client_coordinates(mob)
    }

    /// Copies one Z level into a stable transport-owned map snapshot.
    #[must_use]
    pub fn local_client_map_snapshot(&self, z: i32) -> LocalClientMapSnapshot {
        self.local_client_map_snapshot_for(None, z)
    }

    /// Copies a Z level plus the selected client's screen HUD appearances.
    #[must_use]
    pub fn local_client_map_snapshot_for(
        &self,
        client: Option<DatumId>,
        z: i32,
    ) -> LocalClientMapSnapshot {
        let color_field = FieldName::parse("color").unwrap();
        let mut tiles = self
            .world_turfs
            .iter()
            .filter_map(|(&(x, y, cell_z), &turf)| {
                (cell_z == z)
                    .then(|| {
                        let datum = self.heap.datum(turf).ok()?;
                        let color = datum_field_or_initial(self, turf, &color_field)
                            .ok()
                            .and_then(|value| {
                                (!matches!(value, Value::Null)).then(|| value.to_string())
                            });
                        let occupants: Vec<DatumId> = datum_field_or_initial(
                            self,
                            turf,
                            &FieldName::parse("contents").unwrap(),
                        )
                        .ok()
                        .and_then(|value| match value {
                            Value::List(list) => self.heap.list(list).ok(),
                            _ => None,
                        })
                        .map(|contents| {
                            contents
                                .positions()
                                .filter_map(|(_, value)| match value {
                                    Value::Datum(id) => Some(*id),
                                    _ => None,
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                        let mut draw_datums = Vec::with_capacity(occupants.len() + 1);
                        draw_datums.push(turf);
                        draw_datums.extend(occupants.iter().copied());
                        let mut appearances = draw_datums
                            .into_iter()
                            .enumerate()
                            .filter_map(|(order, datum)| {
                                self.local_client_appearance(datum, 0, &mut HashSet::new())
                                    .map(|appearance| (order, appearance))
                            })
                            .collect::<Vec<_>>();
                        appearances.sort_by(|(left_order, left), (right_order, right)| {
                            left.plane
                                .total_cmp(&right.plane)
                                .then_with(|| left.layer.total_cmp(&right.layer))
                                .then_with(|| left_order.cmp(right_order))
                        });
                        let appearances = appearances
                            .into_iter()
                            .map(|(_, appearance)| appearance)
                            .collect();
                        Some(LocalClientMapTile {
                            x,
                            y,
                            type_path: datum.type_path().as_str().to_owned(),
                            color,
                            occupants,
                            appearances,
                        })
                    })
                    .flatten()
            })
            .collect::<Vec<_>>();
        tiles.sort_by_key(|tile| (tile.y, tile.x));
        let width = tiles.iter().map(|tile| tile.x).max().unwrap_or(0);
        let height = tiles.iter().map(|tile| tile.y).max().unwrap_or(0);
        let mut screen = client
            .and_then(|client| {
                datum_field_or_initial(
                    self,
                    client,
                    &FieldName::parse("screen").expect("client screen field"),
                )
                .ok()
            })
            .and_then(|value| match value {
                Value::List(list) => self.heap.list(list).ok(),
                _ => None,
            })
            .map(|list| {
                list.positions()
                    .filter_map(|(_, value)| match value {
                        Value::Datum(datum) => Some(*datum),
                        _ => None,
                    })
                    .enumerate()
                    .filter_map(|(order, datum)| {
                        let raw_screen_loc = datum_field_or_initial(
                            self,
                            datum,
                            &FieldName::parse("screen_loc").expect("screen_loc field"),
                        )
                        .ok()
                        .and_then(|value| match value {
                            Value::Text(text) => Some(text.to_string()),
                            Value::Null => None,
                            value => Some(value.to_string()),
                        })
                        .unwrap_or_default();
                        let (map_control, screen_loc) = raw_screen_loc
                            .split_once(':')
                            .filter(|(prefix, coordinates)| {
                                !prefix.is_empty()
                                    && prefix.chars().all(|ch| {
                                        ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'
                                    })
                                    && !matches!(
                                        prefix.trim().to_ascii_uppercase().as_str(),
                                        "TOP"
                                            | "BOTTOM"
                                            | "NORTH"
                                            | "SOUTH"
                                            | "LEFT"
                                            | "RIGHT"
                                            | "EAST"
                                            | "WEST"
                                            | "CENTER"
                                    )
                                    && coordinates.contains(',')
                            })
                            .map_or((None, raw_screen_loc.clone()), |(control, coordinates)| {
                                (Some(control.to_owned()), coordinates.to_owned())
                            });
                        self.local_client_appearance(datum, 0, &mut HashSet::new())
                            .map(|appearance| {
                                (
                                    order,
                                    LocalClientScreenAppearance {
                                        map_control,
                                        screen_loc,
                                        insertion: order,
                                        appearance,
                                    },
                                )
                            })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        screen.sort_by(|(left_order, left), (right_order, right)| {
            left.appearance
                .plane
                .total_cmp(&right.appearance.plane)
                .then_with(|| left.appearance.layer.total_cmp(&right.appearance.layer))
                .then_with(|| left_order.cmp(right_order))
        });
        let screen = screen
            .into_iter()
            .map(|(_, appearance)| appearance)
            .collect();
        LocalClientMapSnapshot {
            width,
            height,
            z,
            tiles,
            screen,
        }
    }

    /// Validates and queues a mouse proc on one atom in this client's screen list.
    pub fn queue_local_screen_pointer(
        &mut self,
        module: &Module,
        client: DatumId,
        target_index: u32,
        target_generation: u32,
        event: LocalScreenPointerEvent,
        location: &str,
        params: &str,
    ) -> Result<(), String> {
        let mob = *self
            .local_client_mobs
            .get(&client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        let screen = datum_field_or_initial(
            self,
            client,
            &FieldName::parse("screen").expect("client screen field"),
        )
        .map_err(|error| error.to_string())?;
        let Value::List(screen) = screen else {
            return Err("client screen is not a list".into());
        };
        let target = self
            .heap
            .list(screen)
            .map_err(|error| error.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::Datum(id) => Some(*id),
                _ => None,
            })
            .find(|id| id.index() == target_index && id.generation() == target_generation)
            .ok_or_else(|| "screen target is stale or not owned by session".to_owned())?;
        self.heap.datum(target).map_err(|error| error.to_string())?;
        self.queue_local_atom_pointer(module, client, mob, target, event, location, params)
    }

    /// Validates and queues a click on an atom rendered in the addressed map cell.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_local_map_pointer(
        &mut self,
        module: &Module,
        client: DatumId,
        target_index: u32,
        target_generation: u32,
        x: i32,
        y: i32,
        z: i32,
        control: &str,
        params: &str,
    ) -> Result<(), String> {
        let mob = *self
            .local_client_mobs
            .get(&client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        let target = self
            .heap
            .datum_id_at_index(target_index)
            .filter(|id| id.generation() == target_generation)
            .ok_or_else(|| "map target is stale".to_owned())?;
        let datum = self.heap.datum(target).map_err(|error| error.to_string())?;
        if !is_atom_type_path(datum.type_path()) {
            return Err("map target is not an atom".to_owned());
        }
        let expected = (x as f32, y as f32, z as f32);
        if builtins::datum_coordinates(self, &Value::Datum(target)) != Some(expected) {
            return Err("map target is stale or outside the addressed cell".to_owned());
        }
        if self.turf_at(x, y, z).is_none() {
            return Err("map pointer cell has no materialized turf".to_owned());
        }
        self.queue_local_atom_pointer(
            module,
            client,
            mob,
            target,
            LocalScreenPointerEvent::Click,
            control,
            params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn queue_local_atom_pointer(
        &mut self,
        module: &Module,
        client: DatumId,
        mob: DatumId,
        target: DatumId,
        event: LocalScreenPointerEvent,
        control: &str,
        params: &str,
    ) -> Result<(), String> {
        let target_value = Value::Datum(target);
        let usr = Value::Datum(mob);
        let null_or_text = |value: &str| {
            if value.is_empty() {
                Value::Null
            } else {
                Value::text(value)
            }
        };
        let (receiver, method, arguments) = match event {
            LocalScreenPointerEvent::Click => (
                Value::Datum(client),
                "Click",
                vec![
                    target_value.clone(),
                    Value::Null,
                    null_or_text(control),
                    Value::text(params),
                ],
            ),
            LocalScreenPointerEvent::Entered | LocalScreenPointerEvent::Exited => {
                let location = datum_field_or_initial(
                    self,
                    target,
                    &FieldName::parse("loc").expect("atom loc field"),
                )
                .unwrap_or(Value::Null);
                (
                    target_value.clone(),
                    match event {
                        LocalScreenPointerEvent::Entered => "MouseEntered",
                        LocalScreenPointerEvent::Exited => "MouseExited",
                        LocalScreenPointerEvent::Click => unreachable!(),
                    },
                    vec![location, null_or_text(control), Value::text(params)],
                )
            }
        };
        let caller = ExecutionContext::new(receiver.clone(), usr.clone());
        let resolved = dynamic_call_target_named(module, self, &receiver, method, &caller, false)
            .or_else(|error| {
            // Small fixtures may omit BYOND's built-in `/client/Click`.
            // Preserve direct atom dispatch there while full worlds take
            // OpenDream's client interception path.
            if event != LocalScreenPointerEvent::Click {
                return Err(error);
            }
            let target_context = ExecutionContext::new(target_value.clone(), usr);
            dynamic_call_target_named(module, self, &target_value, "Click", &target_context, false)
        })?;
        let (procedure, context) = resolved;
        let program = module.resolve_procedure(procedure)?;
        let arguments =
            if matches!(event, LocalScreenPointerEvent::Click) && context.src == target_value {
                if program.parameter_names.len() <= 2 {
                    vec![null_or_text(control), Value::text(params)]
                } else {
                    vec![Value::Null, null_or_text(control), Value::text(params)]
                }
            } else {
                arguments
            };
        let frame = make_frame(procedure, program, &arguments, &context);
        schedule_frames(self, vec![frame], 0.0);
        Ok(())
    }

    pub(crate) fn local_client_appearance(
        &self,
        datum: DatumId,
        depth: usize,
        visited: &mut HashSet<DatumId>,
    ) -> Option<LocalClientAppearance> {
        if depth >= 16 || !visited.insert(datum) {
            return None;
        }
        let type_path = self.heap.datum(datum).ok()?.type_path().as_str().to_owned();
        let value =
            |name: &str| datum_field_or_initial(self, datum, &FieldName::parse(name).unwrap()).ok();
        let numeric = |name: &str, fallback: f32| {
            value(name)
                .and_then(|value| value.as_number())
                .unwrap_or(fallback)
        };
        let text = |name: &str| match value(name) {
            Some(Value::Text(text) | Value::File(text)) => Some(text.to_string()),
            Some(Value::Null) | None => None,
            Some(value) => Some(value.to_string()),
        };
        let icon = value("icon").and_then(|value| self.local_client_icon_resource(&value, 0));
        let mut nested = |name: &str| {
            let mut entries = match value(name) {
                Some(Value::List(list)) => self
                    .heap
                    .list(list)
                    .ok()
                    .map(|values| {
                        values
                            .positions()
                            .filter_map(|(_, value)| match value {
                                Value::Datum(datum) => Some(*datum),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                Some(Value::Datum(datum)) => vec![datum],
                _ => Vec::new(),
            }
            .into_iter()
            .enumerate()
            .filter_map(|(order, child)| {
                self.local_client_appearance(child, depth + 1, visited)
                    .map(|appearance| (order, appearance))
            })
            .collect::<Vec<_>>();
            entries.sort_by(|(left_order, left), (right_order, right)| {
                left.plane
                    .total_cmp(&right.plane)
                    .then_with(|| left.layer.total_cmp(&right.layer))
                    .then_with(|| left_order.cmp(right_order))
            });
            entries
                .into_iter()
                .map(|(_, appearance)| appearance)
                .collect()
        };
        let underlays = nested("underlays");
        let overlays = nested("overlays");
        visited.remove(&datum);
        Some(LocalClientAppearance {
            datum,
            type_path,
            icon,
            icon_state: text("icon_state"),
            dir: numeric("dir", 2.0) as i32,
            layer: numeric("layer", 0.0),
            plane: numeric("plane", 0.0),
            appearance_flags: numeric("appearance_flags", 0.0) as i32,
            mouse_opacity: numeric("mouse_opacity", 1.0) as i32,
            pixel_x: numeric("pixel_x", 0.0),
            pixel_y: numeric("pixel_y", 0.0),
            pixel_w: numeric("pixel_w", 0.0),
            pixel_z: numeric("pixel_z", 0.0),
            color: text("color"),
            alpha: numeric("alpha", 255.0),
            maptext: text("maptext"),
            maptext_width: numeric("maptext_width", 0.0),
            maptext_height: numeric("maptext_height", 0.0),
            maptext_x: numeric("maptext_x", 0.0),
            maptext_y: numeric("maptext_y", 0.0),
            underlays,
            overlays,
        })
    }

    pub(crate) fn local_client_icon_resource(&self, value: &Value, depth: usize) -> Option<String> {
        if depth >= 16 {
            return None;
        }
        match value {
            Value::File(path) | Value::Text(path) => Some(path.to_string()),
            Value::Datum(icon) => {
                let datum = self.heap.datum(*icon).ok()?;
                let path = datum.type_path().as_str();
                if path != "/icon" && !path.starts_with("/icon/") {
                    return None;
                }
                let backing =
                    datum_field_or_initial(self, *icon, &FieldName::parse("icon").unwrap()).ok()?;
                self.local_client_icon_resource(&backing, depth + 1)
            }
            _ => None,
        }
    }

    pub(crate) fn local_client_coordinates(&self, mob: DatumId) -> Result<(i32, i32, i32), String> {
        let loc = FieldName::parse("loc").unwrap();
        let Value::Datum(turf) =
            datum_field_or_initial(self, mob, &loc).map_err(|error| error.to_string())?
        else {
            return Err("controlled mob is not located on a turf".to_owned());
        };
        self.world_turfs
            .iter()
            .find_map(|(coordinate, candidate)| (*candidate == turf).then_some(*coordinate))
            .ok_or_else(|| {
                "controlled mob turf is absent from the authoritative world index".to_owned()
            })
    }

    pub(crate) fn environment_override(&self, name: &str) -> Option<&Option<Value>> {
        self.environment_overrides.get(name)
    }

    pub(crate) fn set_environment_override(&mut self, name: String, value: Option<Value>) {
        self.environment_overrides.insert(name, value);
    }

    pub(crate) fn reset_external_timer(&mut self, name: String) {
        self.external_timers.insert(name, Instant::now());
    }

    pub(crate) fn external_timer_milliseconds(&self, name: &str) -> f64 {
        self.external_timers
            .get(name)
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
    }

    pub(crate) fn external_timer_microseconds(&self, name: &str) -> f64 {
        self.external_timers
            .get(name)
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1_000_000.0)
    }

    pub(crate) fn enqueue_iconforge_job(&mut self, result: String) -> String {
        self.iconforge_next_job = self.iconforge_next_job.saturating_add(1);
        let id = format!("dream64-iconforge-{}", self.iconforge_next_job);
        self.iconforge_jobs.insert(id.clone(), (false, result));
        id
    }

    pub(crate) fn poll_iconforge_job(&mut self, id: &str) -> Option<String> {
        let (polled, _) = self.iconforge_jobs.get(id)?;
        if !*polled {
            // Jobs are launched concurrently. One pending observation advances
            // the deterministic headless worker so the caller's next sweep can
            // collect every completed result without one sleep per job.
            for (polled, _) in self.iconforge_jobs.values_mut() {
                *polled = true;
            }
            return Some("NO RESULTS YET".to_owned());
        }
        self.iconforge_jobs.remove(id).map(|(_, result)| result)
    }

    pub(crate) fn load_iconforge_gags_config(&mut self, path: String, source: PathBuf) {
        self.iconforge_gags_configs.insert(path, source);
    }

    pub(crate) fn has_iconforge_gags_config(&self, path: &str) -> bool {
        self.iconforge_gags_configs.contains_key(path)
    }

    pub(crate) fn iconforge_gags_source(&self, path: &str) -> Option<&std::path::Path> {
        self.iconforge_gags_configs.get(path).map(PathBuf::as_path)
    }

    pub(crate) fn enqueue_sql_job(&mut self, result: String) -> String {
        self.sql_next_job = self.sql_next_job.saturating_add(1);
        let id = format!("dream64-sql-{}", self.sql_next_job);
        self.sql_jobs.insert(id.clone(), (false, result));
        id
    }

    pub(crate) fn poll_sql_job(&mut self, id: &str) -> Option<String> {
        let (polled, _) = self.sql_jobs.get(id)?;
        if !*polled {
            for (polled, _) in self.sql_jobs.values_mut() {
                *polled = true;
            }
            return Some("NO RESULTS YET".to_owned());
        }
        self.sql_jobs.remove(id).map(|(_, result)| result)
    }

    pub(crate) fn is_associative_list(&self, list: ListId) -> bool {
        self.associative_lists.contains(&list)
    }

    pub(crate) fn mark_associative_list(&mut self, list: ListId) {
        self.associative_lists.insert(list);
    }

    pub(crate) fn refresh_vars_proxy(&mut self, list: ListId) -> Result<(), String> {
        let Some(datum) = self.datum_vars_proxies.get(&list).copied() else {
            return Ok(());
        };
        let keys = self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, key)| key.clone())
            .collect::<Vec<_>>();
        for key in keys {
            let Value::Text(name) = &key else {
                continue;
            };
            let field = FieldName::parse(name).map_err(|error| error.to_string())?;
            let value = if let Some(value) = lazy_atom_list_field(self, datum, &field)? {
                value
            } else {
                datum_shared_storage(self, datum, &field)
                    .and_then(|storage| self.global(&storage).cloned())
                    .or_else(|| datum_field_or_initial(self, datum, &field).ok())
                    .unwrap_or(Value::Null)
            };
            self.heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .set_key(key, value);
        }
        Ok(())
    }

    /// Reads a persistent runtime global.
    #[must_use]
    pub fn global(&self, name: &FieldName) -> Option<&Value> {
        self.globals.get(name)
    }

    /// Inserts or replaces a persistent runtime global.
    pub fn set_global(&mut self, name: FieldName, value: Value) -> Option<Value> {
        self.assert_owner_thread();
        let is_new = self.globals.get(&name).is_none();
        let previous = self.globals.insert(name.clone(), value);
        if is_new
            && let Some(list) = self.global_vars_proxy
            && let Ok(values) = self.heap.list_mut(list)
        {
            values.add(Value::text(name.as_str()));
        }
        previous
    }

    /// Records a declaration-time global/static value used by `initial()`.
    pub fn set_initial_global(&mut self, name: FieldName, value: Value) -> Option<Value> {
        self.initial_globals.insert(name, value)
    }

    /// Deletes a persistent runtime global.
    pub fn delete_global(&mut self, name: &FieldName) -> Option<Value> {
        self.globals.remove(name)
    }

    /// Replaces the canonical type catalog used by `typesof()`.
    pub fn set_type_paths(&mut self, paths: impl IntoIterator<Item = TypePath>) {
        self.type_paths = Arc::new(paths.into_iter().collect());
        self.clear_effective_initial_value_cache();
    }

    /// Replaces the canonical type catalog used by `typesof()` with a shared
    /// immutable catalog.
    ///
    /// Runtime images use this to avoid cloning a project's complete object
    /// tree for every dynamically evaluated initializer.
    pub fn set_shared_type_paths(&mut self, paths: Arc<std::collections::BTreeSet<TypePath>>) {
        self.type_paths = paths;
        self.clear_effective_initial_value_cache();
    }

    /// Iterates the canonical type catalog in lexical path order.
    pub fn type_paths(&self) -> impl Iterator<Item = &TypePath> {
        self.type_paths.iter()
    }

    /// Replaces the runtime type-parent catalog used by subtype and `parent_type` lookups.
    pub fn set_type_parents(&mut self, parents: BTreeMap<TypePath, Option<TypePath>>) {
        self.type_intervals = Arc::new(build_type_intervals(&parents));
        self.type_parents = Arc::new(parents);
        self.dynamic_receiver_targets.clear();
        self.dynamic_callsite_targets.clear();
        self.instance_initializer_plans.clear();
        self.clear_effective_initial_value_cache();
    }

    /// Replaces the runtime type-parent catalog with shared immutable metadata.
    pub fn set_shared_type_parents(&mut self, parents: Arc<BTreeMap<TypePath, Option<TypePath>>>) {
        self.type_intervals = Arc::new(build_type_intervals(&parents));
        self.type_parents = parents;
        self.dynamic_receiver_targets.clear();
        self.dynamic_callsite_targets.clear();
        self.instance_initializer_plans.clear();
        self.clear_effective_initial_value_cache();
    }

    pub(crate) fn subtype_interval(&self, path: &TypePath) -> Option<(u32, u32)> {
        self.type_intervals.get(path).copied()
    }

    /// Replaces effective compile-time initial field values for every runtime type.
    pub fn set_initial_values(&mut self, values: BTreeMap<TypePath, BTreeMap<FieldName, Value>>) {
        self.initial_values = Arc::new(values);
        self.rebuild_initial_value_roots();
        self.clear_effective_initial_value_cache();
    }

    /// Replaces effective initial values with shared immutable metadata.
    pub fn set_shared_initial_values(
        &mut self,
        values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,
    ) {
        self.initial_values = values;
        self.rebuild_initial_value_roots();
        self.clear_effective_initial_value_cache();
    }

    pub(crate) fn rebuild_initial_value_roots(&mut self) {
        let mut datum_roots = Vec::new();
        let mut list_roots = Vec::new();
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.initial_values.values().flat_map(BTreeMap::values),
        );
        datum_roots.sort_unstable();
        datum_roots.dedup();
        list_roots.sort_unstable();
        list_roots.dedup();
        self.initial_value_datum_roots = Arc::from(datum_roots);
        self.initial_value_list_roots = Arc::from(list_roots);
    }

    /// Installs inherited reflection names for owner-qualified shared fields.
    pub fn set_shared_fields(
        &mut self,
        fields: Arc<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>>,
    ) {
        self.shared_fields = fields;
    }

    /// Installs direct per-type initializer programs used by runtime `new`.
    pub fn set_instance_initializers(
        &mut self,
        initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
        module: Option<Arc<Module>>,
    ) {
        self.instance_initializers = initializers;
        self.instance_initializer_module = module;
        self.instance_initializer_plans.clear();
        self.clear_initial_field_value_cache();
    }

    /// Replaces runtime-new initializer metadata and returns the previous catalog.
    pub fn replace_instance_initializers(
        &mut self,
        initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
    ) -> Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>> {
        let previous = std::mem::replace(&mut self.instance_initializers, initializers);
        self.instance_initializer_plans.clear();
        self.clear_initial_field_value_cache();
        previous
    }

    /// Sets the project root used by BYOND filesystem procedures such as `fexists()`.
    pub fn set_project_root(&mut self, root: PathBuf) {
        self.project_root = Some(Arc::new(root));
    }

    /// Installs immutable artifact-time measurements for project DMM resources.
    pub fn set_dmm_measurements(&mut self, measurements: Arc<BTreeMap<String, DmmMeasurement>>) {
        self.dmm_measurements = measurements;
    }

    /// Returns the immutable project DMM measurement catalog.
    #[must_use]
    pub fn dmm_measurements(&self) -> Arc<BTreeMap<String, DmmMeasurement>> {
        Arc::clone(&self.dmm_measurements)
    }

    /// Installs immutable artifact-time full parsed-map products.
    pub fn set_parsed_dmm_cache(&mut self, cache: Arc<BTreeMap<String, ParsedDmm>>) {
        self.parsed_dmm_cache = cache;
    }

    /// Returns the immutable full parsed-map catalog.
    #[must_use]
    pub fn parsed_dmm_cache(&self) -> Arc<BTreeMap<String, ParsedDmm>> {
        Arc::clone(&self.parsed_dmm_cache)
    }

    /// Returns a type's runtime parent when the catalog contains that type.
    #[must_use]
    pub fn type_parent(&self, path: &TypePath) -> Option<&TypePath> {
        self.type_parents.get(path).and_then(Option::as_ref)
    }

    /// Returns one effective compile-time initial value when available.
    #[must_use]
    pub fn initial_value(&self, path: &TypePath, field: &FieldName) -> Option<&Value> {
        let mut current = Some(path);
        while let Some(path) = current {
            if let Some(value) = self
                .initial_values
                .get(path)
                .and_then(|fields| fields.get(field))
            {
                return Some(value);
            }
            current = self.type_parent(path);
        }
        None
    }

    pub(crate) fn inherited_initial_values(&self, path: &TypePath) -> BTreeMap<FieldName, Value> {
        let mut hierarchy = Vec::new();
        let mut current = Some(path);
        while let Some(path) = current {
            hierarchy.push(path);
            current = self.type_parent(path);
        }
        let mut values = BTreeMap::new();
        for path in hierarchy.into_iter().rev() {
            if let Some(direct) = self.initial_values.get(path) {
                values.extend(direct.clone());
            }
        }
        values
    }

    pub(crate) fn clear_effective_initial_value_cache(&mut self) {
        self.effective_initial_value_cache.get_mut().clear();
        self.effective_initial_value_cache_entries.set(0);
        self.clear_initial_field_value_cache();
    }

    pub(crate) fn clear_initial_field_value_cache(&mut self) {
        self.initial_field_value_cache.clear();
        self.initial_field_value_cache_entries = 0;
    }

    pub(crate) fn effective_initial_value(
        &self,
        path: &TypePath,
        field: &FieldName,
    ) -> Option<Value> {
        let cache = self.effective_initial_value_cache.borrow();
        if let Some(value) = cache.get(path).and_then(|fields| fields.get(field)) {
            return value.clone();
        }
        drop(cache);

        let value = self
            .initial_value(path, field)
            .or_else(|| engine_root_initial_value(self, path, field))
            .cloned()
            .or_else(|| engine_builtin_initial_value(path, field));
        let entry_count = self.effective_initial_value_cache_entries.get();
        if entry_count < MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES {
            let mut cache = self.effective_initial_value_cache.borrow_mut();
            let fields = cache.entry(path.clone()).or_default();
            if fields.len() < MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE {
                fields.insert(field.clone(), value.clone());
                self.effective_initial_value_cache_entries
                    .set(entry_count + 1);
            }
        }
        value
    }

    /// Seeds effective project and engine defaults on a datum allocated by a
    /// native constructor (`image()`, `icon()`, `sound()`, and peers). Native
    /// constructors historically allocated raw heap datums, bypassing the
    /// inherited `/datum` fields that ordinary `new` installs.
    pub(crate) fn seed_native_datum_defaults(
        &mut self,
        datum: DatumId,
        path: &TypePath,
    ) -> Result<(), String> {
        let mut defaults = engine_builtin_initial_fields(path);
        defaults.extend(self.inherited_initial_values(path));
        for (field, value) in defaults {
            if self.heap.datum_field(datum, &field).is_err() {
                self.heap
                    .set_datum_field(datum, field, value)
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }

    /// Returns the project root used for relative filesystem paths.
    #[must_use]
    pub fn project_root(&self) -> Option<&std::path::Path> {
        self.project_root.as_deref().map(PathBuf::as_path)
    }

    /// Returns the shared immutable runtime type catalog.
    #[must_use]
    pub fn shared_type_paths(&self) -> Arc<BTreeSet<TypePath>> {
        Arc::clone(&self.type_paths)
    }

    /// Returns the shared immutable runtime inheritance catalog.
    #[must_use]
    pub fn shared_type_parents(&self) -> Arc<BTreeMap<TypePath, Option<TypePath>>> {
        Arc::clone(&self.type_parents)
    }

    /// Returns the shared immutable direct initial-value catalog.
    #[must_use]
    pub fn shared_initial_values(&self) -> Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>> {
        Arc::clone(&self.initial_values)
    }

    /// Returns the shared immutable reflection field catalog.
    #[must_use]
    pub fn shared_fields(&self) -> Arc<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>> {
        Arc::clone(&self.shared_fields)
    }

    /// Returns linked per-instance initializer actions and their portable module.
    #[must_use]
    pub fn shared_instance_initializers(
        &self,
    ) -> (
        Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
        Option<Arc<Module>>,
    ) {
        (
            Arc::clone(&self.instance_initializers),
            self.instance_initializer_module.clone(),
        )
    }

    /// Iterates globals in canonical field-name order for snapshots.
    pub fn globals(&self) -> impl Iterator<Item = (&FieldName, &Value)> {
        self.globals.iter()
    }

    /// Returns the current deterministic scheduler tick.
    #[must_use]
    pub const fn scheduler_tick(&self) -> u64 {
        self.scheduler_tick
    }

    /// Replaces the per-world random stream seed used by DM `rand()`,
    /// `pick()`, `prob()`, `roll()`, and related engine fallbacks.
    ///
    /// Hosts call this once for every real process launch so structural caches
    /// never turn independent rounds into repetitions of one deterministic
    /// startup stream. DM's explicit `rand_seed()` can still replace it later.
    pub const fn reseed_random(&mut self, seed: u64) {
        self.random_state = seed;
    }

    /// Returns the current per-world random stream state.
    #[must_use]
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }

    /// Returns the number of suspended or spawned tasks awaiting dispatch.
    #[must_use]
    pub fn scheduled_task_count(&self) -> usize {
        self.scheduled_spawns.len() + self.pending_local_prompts.len()
    }

    /// Returns allocation-neutral suspended-frame metrics for runtime benchmarks.
    #[must_use]
    pub fn continuation_metrics(&self) -> ContinuationMetrics {
        let mut metrics = ContinuationMetrics {
            frame_header_bytes: std::mem::size_of::<CallFrame>(),
            rare_inline_bytes_avoided: std::mem::size_of::<Option<Value>>()
                + std::mem::size_of::<Option<Box<CallFrame>>>()
                + std::mem::size_of::<Option<NumericExecutionState>>(),
            ..ContinuationMetrics::default()
        };
        for continuation in self
            .scheduled_spawns
            .iter()
            .chain(self.scheduler_inflight.iter())
            .map(|spawn| &spawn.frames)
        {
            metrics.continuations += 1;
            metrics.frames += continuation.len();
            for frame in continuation {
                metrics.cold_frames += usize::from(frame.cold.is_some());
                metrics.retained_values += frame.locals.len()
                    + frame.stack.len()
                    + frame.arguments.len()
                    + frame.pending_argument_roots().len()
                    + frame.retained_call_roots().len()
                    + 3
                    + usize::from(frame.args_list.is_some())
                    + usize::from(frame.caller_result_override().is_some());
            }
        }
        metrics
    }

    /// Returns static-field quickening counters for this owner-thread VM state.
    #[must_use]
    pub const fn declared_field_quickening_metrics(&self) -> DeclaredFieldQuickeningMetrics {
        self.declared_field_quickening
    }

    /// Preferred name for the static-field quickening telemetry.
    #[must_use]
    pub const fn static_field_quickening_metrics(&self) -> DeclaredFieldQuickeningMetrics {
        self.declared_field_quickening
    }

    /// Formats a bounded snapshot of suspended DM continuations for shutdown diagnostics.
    ///
    /// This performs no sampling or bookkeeping while bytecode executes. Hosts call it only
    /// after a scheduler bound terminates, so startup hot loops pay no telemetry overhead.
    #[must_use]
    pub fn bounded_scheduler_progress(&self, module: &Module) -> Vec<String> {
        let mut tasks = self.scheduled_spawns.iter().collect::<Vec<_>>();
        tasks.sort_unstable_by_key(|task| (task.due_tick, task.sequence));
        let mut lines = Vec::new();
        for (task_index, task) in tasks.into_iter().take(4).enumerate() {
            for (frame_index, frame) in task.frames.iter().rev().take(8).enumerate() {
                let path = module
                    .procedure_path(frame.procedure)
                    .unwrap_or("<missing>");
                let program = module.procedure(frame.procedure);
                let source = program
                    .and_then(|program| program.source_spans.get(frame.instruction))
                    .map_or_else(|| "<missing>".to_owned(), |span| format!("{span:?}"));
                let parameters = program.map_or_else(String::new, |program| {
                    program
                        .parameter_names
                        .iter()
                        .zip(frame.locals.iter())
                        .filter(|(name, _)| !name.is_empty())
                        .take(4)
                        .map(|(name, value)| {
                            format!("{name}={}", bounded_progress_value(value, &self.heap))
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                });
                let map_progress = map_loader_progress(frame, path, &self.heap);
                lines.push(format!(
                    "task={task_index} continuation={} due_tick={} sequence={} depth={} frame={} procedure={} instruction={} source={} stack={} locals={} parameters=[{}]{}",
                    task.frames.id.get(),
                    task.due_tick,
                    task.sequence,
                    task.frames.len(),
                    frame_index,
                    path,
                    frame.instruction,
                    source,
                    frame.stack.len(),
                    frame.locals.len(),
                    parameters,
                    map_progress,
                ));
            }
        }
        lines
    }

    /// Releases values returned across completed host execution calls.
    ///
    /// Callers that have consumed those results can use this before the next
    /// collection cycle so temporary datum/list identities are not retained
    /// for the lifetime of the server.
    pub fn release_host_value_roots(&mut self) -> usize {
        let released = self.host_value_roots.len();
        self.host_value_roots.clear();
        released
    }

    /// Forces one full heap collection at a host-owned quiescent boundary.
    ///
    /// Startup calls this only after VM execution has returned all active
    /// frames to the scheduler-owned queues, so the normal global, scheduler,
    /// client, world, and host roots are complete without an extra frame set.
    pub fn compact_quiescent_heap(&mut self) -> QuiescentHeapCompaction {
        let before_datums = self.heap.live_datum_count();
        let before_lists = self.heap.live_list_count();
        let started = Instant::now();
        self.next_list_collection = 1;
        self.maybe_collect_unreachable_lists(&[]);
        QuiescentHeapCompaction {
            reclaimed_datums: before_datums.saturating_sub(self.heap.live_datum_count()),
            reclaimed_lists: before_lists.saturating_sub(self.heap.live_list_count()),
            elapsed: started.elapsed(),
        }
    }

    /// Returns the earliest tick at which pending scheduler work is due.
    #[must_use]
    pub fn next_scheduled_tick(&self) -> Option<u64> {
        self.scheduled_spawns
            .iter()
            .map(|task| task.due_tick)
            .chain(self.native_walks.values().map(|walk| walk.due_tick))
            .min()
    }

    pub(crate) fn maybe_collect_unreachable_lists(&mut self, active_frames: &[CallFrame]) {
        if self.next_list_collection == 0 {
            self.next_list_collection = MINIMUM_HEAP_COLLECTION_GROWTH;
        }
        let before_lists = self.heap.live_list_count();
        let before_datums = self.heap.live_datum_count();
        let before = before_lists.saturating_add(before_datums);
        if before < self.next_list_collection {
            return;
        }
        let collection_started = Instant::now();

        let mut datum_roots = Vec::new();
        let mut list_roots = Vec::new();
        extend_heap_root_ids(&mut datum_roots, &mut list_roots, self.globals.values());
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.initial_globals.values(),
        );
        datum_roots.extend_from_slice(&self.initial_value_datum_roots);
        list_roots.extend_from_slice(&self.initial_value_list_roots);
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.procedure_static_locals
                .values()
                .flat_map(|slots| slots.values()),
        );
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.savefiles
                .values()
                .flat_map(|savefile| savefile.entries.values()),
        );
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.environment_overrides.values().flatten(),
        );
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.last_animation_target.iter(),
        );
        extend_heap_root_ids(&mut datum_roots, &mut list_roots, &self.host_value_roots);
        datum_roots.extend(self.initial_prototypes.values().copied());
        datum_roots.extend(self.savefiles.keys().copied());
        datum_roots.extend(
            self.savefile_entries
                .iter()
                .flat_map(|(entry, (savefile, _))| [*entry, *savefile]),
        );
        datum_roots.extend(self.deleting_datums.iter().copied());
        datum_roots.extend(self.client_sessions.keys().copied());
        datum_roots.extend(self.local_client_mobs.keys().copied());
        datum_roots.extend(self.local_client_mobs.values().copied());
        datum_roots.extend(self.compact_default_datums.iter().copied());
        datum_roots.extend(self.world_turfs.values().copied());
        datum_roots.extend(self.world_areas.values().copied());
        datum_roots.extend(self.default_world_area);
        datum_roots.extend(self.native_walks.keys().copied());
        datum_roots.extend(self.native_walks.values().filter_map(NativeWalk::target));
        list_roots.extend(self.global_vars_proxy);
        for (datum, list) in self
            .datum_vars_by_datum
            .iter()
            .filter(|(datum, _)| self.heap.datum(**datum).is_ok())
        {
            datum_roots.push(*datum);
            list_roots.push(*list);
        }
        for prompt in self.pending_local_prompts.values() {
            datum_roots.push(prompt.client);
            extend_heap_root_ids(&mut datum_roots, &mut list_roots, &prompt.choices);
        }

        let mut add_frame_roots = |frame: &CallFrame| {
            // Engine-owned post-return continuations are suspended frames, not
            // disposable metadata. A collection may run while their child is
            // active, so root the complete boxed continuation chain exactly as
            // we do for ordinary scheduler frames.
            let mut frame = Some(frame);
            while let Some(current) = frame {
                extend_heap_root_ids(&mut datum_roots, &mut list_roots, &current.locals);
                extend_heap_root_ids(&mut datum_roots, &mut list_roots, &current.stack);
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    std::iter::once(&current.result),
                );
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    std::iter::once(&current.src),
                );
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    std::iter::once(&current.usr),
                );
                extend_heap_root_ids(&mut datum_roots, &mut list_roots, &current.arguments);
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    current.pending_argument_roots(),
                );
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    current.retained_call_roots(),
                );
                if let Some(tgm) = current.cold().and_then(|cold| cold.tgm_load.as_ref()) {
                    extend_heap_root_ids(&mut datum_roots, &mut list_roots, tgm.roots());
                }
                if let Some(scan) = current.cold().and_then(|cold| cold.ruin_scan.as_ref()) {
                    datum_roots.extend(scan.turfs.iter().copied());
                    extend_heap_root_ids(&mut datum_roots, &mut list_roots, &scan.areas);
                }
                list_roots.extend(current.args_list);
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    current.caller_result_override().into_iter(),
                );
                frame = current.engine_post_return();
            }
        };
        for frame in active_frames {
            add_frame_roots(frame);
        }
        for spawn in &self.scheduled_spawns {
            for frame in &spawn.frames {
                add_frame_roots(frame);
            }
        }
        for spawn in &self.scheduler_inflight {
            for frame in &spawn.frames {
                add_frame_roots(frame);
            }
        }
        for prompt in self.pending_local_prompts.values() {
            match &prompt.continuation {
                PendingPromptContinuation::Frames(frames) => {
                    for frame in frames {
                        add_frame_roots(frame);
                    }
                }
                PendingPromptContinuation::Verb(invocation) => {
                    add_frame_roots(&invocation.frame);
                }
            }
        }

        let collection = self
            .heap
            .collect_unreachable_values_from_ids_with_stats(&datum_roots, &list_roots);
        let reclaimed_datums = collection.reclaimed_datums;
        let reclaimed_lists = collection.reclaimed_lists;
        let heap = &self.heap;
        self.associative_lists
            .retain(|list| heap.list(*list).is_ok());
        self.reference_lists.retain(|list| heap.list(*list).is_ok());
        self.datum_vars_proxies
            .retain(|list, datum| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.datum_vars_by_datum
            .retain(|datum, list| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.contents_owners
            .retain(|list, datum| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.vis_contents_owners
            .retain(|list, datum| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.vis_locs_owners
            .retain(|list, datum| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.compact_default_datums
            .retain(|datum| heap.datum(*datum).is_ok());
        self.native_walks.retain(|movable, walk| {
            heap.datum(*movable).is_ok()
                && walk
                    .target()
                    .is_none_or(|target| heap.datum(target).is_ok())
        });
        if self
            .global_vars_proxy
            .is_some_and(|list| heap.list(list).is_err())
        {
            self.global_vars_proxy = None;
        }

        let after_lists = self.heap.live_list_count();
        let after_datums = self.heap.live_datum_count();
        let after = after_lists.saturating_add(after_datums);
        let reclaimed = reclaimed_datums.saturating_add(reclaimed_lists);
        let growth = adaptive_heap_collection_growth(after, reclaimed);
        self.next_list_collection = after.saturating_add(growth);
        if boot_trace_enabled() || boot_dashboard_enabled() {
            let elapsed_ms = collection_started.elapsed().as_millis();
            let lists = collection.list_storage;
            let datum_storage = collection.datum_storage;
            let datums = collection.datum_arena;
            let list_arena = collection.list_arena;
            eprintln!(
                "boot-vm: heap-gc datums_before={before_datums} datums_after={after_datums} datums_reclaimed={reclaimed_datums} lists_before={before_lists} lists_after={after_lists} lists_reclaimed={reclaimed_lists} growth={growth} next={} elapsed_ms={elapsed_ms} list_storage_allocated={} list_payload_len={} list_payload_cap={} list_order_len={} list_order_cap={} list_prefix_retained={} list_prefix_compacted={} list_prefix_entries_compacted={} list_vectors_shrunk={} list_capacity_bytes_reclaimed={} list_shared_shrink_candidates={} list_assoc_indexes={} list_assoc_index_len={} list_assoc_index_cap={} list_assoc_index_ratio_bins={:?} list_assoc_indexes_shrunk={} list_assoc_index_cap_reclaimed={} list_assoc_index_bytes_reclaimed={} list_remove_indexes={} list_remove_key_len={} list_remove_key_cap={} list_remove_position_len={} list_remove_position_cap={} list_remove_removed_len={} list_remove_removed_cap={} list_remove_indexes_dropped={} list_shared_derived_candidates={} datum_field_len={} datum_field_cap={} datum_shared_name_datums={} datum_shared_name_logical_slots={} datum_shared_name_layouts={} datum_shared_name_physical_slots={} datum_shared_name_bytes_saved={} datum_field_vectors_shrunk={} datum_capacity_bytes_reclaimed={} datum_field_indexes={} datum_field_index_len={} datum_field_index_cap={} datum_field_index_ratio_bins={:?} datum_field_indexes_shrunk={} datum_field_index_cap_reclaimed={} datum_field_index_bytes_reclaimed={} datum_field_indexes_deduplicated={} datum_field_index_dedupe_bytes_reclaimed={} datum_field_physical_indexes={} datum_field_physical_index_len={} datum_field_physical_index_cap={} datum_field_index_fingerprint_collisions={} datum_field_index_fingerprints_computed={} datum_field_index_pointer_cache_hits={} datum_field_index_exact_layout_comparisons={} datum_arena_live={} datum_arena_slots={} datum_arena_free={} datum_arena_chunks={} datum_arena_reserved={} list_arena_live={} list_arena_slots={} list_arena_free={} list_arena_chunks={} list_arena_reserved={}",
                self.next_list_collection,
                lists.allocated_lists,
                lists.payload_len,
                lists.payload_capacity,
                lists.order_len,
                lists.order_capacity,
                lists.prefix_retained,
                lists.compacted_lists,
                lists.compacted_prefix_entries,
                lists.shrunk_vectors,
                lists.reclaimed_capacity_bytes,
                lists.shared_shrink_candidates,
                lists.associative_indexes,
                lists.associative_index_len,
                lists.associative_index_capacity,
                lists.associative_index_ratio_bins,
                lists.shrunk_associative_indexes,
                lists.reclaimed_associative_index_capacity,
                lists.reclaimed_associative_index_bytes,
                lists.positional_remove_indexes,
                lists.positional_remove_key_len,
                lists.positional_remove_key_capacity,
                lists.positional_remove_position_len,
                lists.positional_remove_position_capacity,
                lists.positional_remove_removed_len,
                lists.positional_remove_removed_capacity,
                lists.dropped_positional_remove_indexes,
                lists.shared_derived_index_candidates,
                datum_storage.field_len,
                datum_storage.field_capacity,
                datum_storage.shared_field_name_datums,
                datum_storage.shared_field_name_logical_slots,
                datum_storage.shared_field_name_layouts,
                datum_storage.shared_field_name_physical_slots,
                datum_storage.shared_field_name_bytes_saved,
                datum_storage.shrunk_field_vectors,
                datum_storage.reclaimed_capacity_bytes,
                datum_storage.field_indexes,
                datum_storage.field_index_len,
                datum_storage.field_index_capacity,
                datum_storage.field_index_ratio_bins,
                datum_storage.shrunk_field_indexes,
                datum_storage.reclaimed_field_index_capacity,
                datum_storage.reclaimed_field_index_bytes,
                datum_storage.deduplicated_field_indexes,
                datum_storage.deduplicated_field_index_bytes,
                datum_storage.physical_field_indexes,
                datum_storage.physical_field_index_len,
                datum_storage.physical_field_index_capacity,
                datum_storage.field_index_fingerprint_collisions,
                datum_storage.field_index_fingerprints_computed,
                datum_storage.field_index_pointer_cache_hits,
                datum_storage.field_index_exact_layout_comparisons,
                datums.live,
                datums.slots,
                datums.free,
                datums.chunks,
                datums.reserved,
                list_arena.live,
                list_arena.slots,
                list_arena.free,
                list_arena.chunks,
                list_arena.reserved,
            );
            if std::env::var_os("DREAM64_PROFILE_MEMORY").is_some() {
                let mut counts = HashMap::<TypePath, usize>::new();
                for (_, datum) in self.heap.datums() {
                    *counts.entry(datum.type_path().clone()).or_default() += 1;
                }
                let mut counts = counts.into_iter().collect::<Vec<_>>();
                counts.sort_unstable_by(|(left_path, left_count), (right_path, right_count)| {
                    right_count
                        .cmp(left_count)
                        .then_with(|| left_path.as_str().cmp(right_path.as_str()))
                });
                let summary = counts
                    .into_iter()
                    .take(20)
                    .map(|(path, count)| format!("{}={count}", path.as_str()))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!("boot-vm: heap-type-census {summary}");
            }
        }
    }
}
fn bounded_progress_value(value: &Value, heap: &ValueHeap) -> String {
    match value {
        Value::Text(text) => {
            let truncated = text.chars().count() > 48;
            let mut text = text.chars().take(48).collect::<String>();
            if truncated {
                text.push('…');
            }
            format!("{text:?}")
        }
        Value::List(list) => heap.list(*list).map_or_else(
            |_| "list(stale)".to_owned(),
            |values| format!("list(len={})", values.len()),
        ),
        Value::Datum(datum) => heap.datum(*datum).map_or_else(
            |_| "datum(stale)".to_owned(),
            |value| format!("datum({})", value.type_path()),
        ),
        _ => value.to_string(),
    }
}

fn map_loader_progress(frame: &CallFrame, path: &str, heap: &ValueHeap) -> String {
    if !path.contains("/datum/parsed_map/proc/") {
        return String::new();
    }
    let Value::Datum(src) = frame.src else {
        return String::new();
    };
    let fields = ["map_format", "grid_models", "modelCache", "bounds"]
        .into_iter()
        .filter_map(|name| {
            let field = FieldName::parse(name).ok()?;
            let value = heap.datum_field(src, &field).ok()?;
            Some(format!("{name}={}", bounded_progress_value(value, heap)))
        })
        .collect::<Vec<_>>();
    (!fields.is_empty())
        .then(|| format!(" map=[{}]", fields.join(",")))
        .unwrap_or_default()
}

fn normalize_client_command_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn split_client_command(command: &str) -> Result<(&str, &str), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("client command is empty".to_owned());
    }
    Ok(command
        .split_once(char::is_whitespace)
        .map_or((command, ""), |(name, arguments)| (name, arguments.trim())))
}

fn parse_client_command_arguments(arguments: &str) -> Result<Vec<String>, String> {
    let mut parsed = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut escaped = false;
    for ch in arguments.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if quoted && ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            quoted = !quoted;
            continue;
        }
        if ch.is_whitespace() && !quoted {
            if !current.is_empty() {
                parsed.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(ch);
    }
    if escaped {
        current.push('\\');
    }
    if quoted {
        return Err("client command has an unterminated quote".to_owned());
    }
    if !current.is_empty() {
        parsed.push(current);
    }
    Ok(parsed)
}

pub(crate) const MINIMUM_HEAP_COLLECTION_GROWTH: usize = 65_536;
// Keep the low-yield window bounded tightly enough for production hosts with
// small Windows commit limits. Boot194 exhausted commit after this window
// expanded to 836,706 live identities, before the next reachability pass.
pub(crate) const MAXIMUM_LOW_YIELD_COLLECTION_GROWTH: usize = 262_144;
pub(crate) const MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH: usize = 262_144;
pub(crate) const MAXIMUM_HIGH_YIELD_COLLECTION_GROWTH: usize = 262_144;

/// Chooses how many additional heap identities may be allocated before the
/// next full reachability pass.
///
/// A large, mostly-live production heap is expensive to rescan and receives a
/// 25% growth window after a pass that reclaimed at most 5% of the identities
/// it visited. More productive passes tighten the window because they indicate
/// that transient allocation pressure is building. Absolute caps bound both
/// transient garbage and the live-object growth between collections.
pub(crate) fn adaptive_heap_collection_growth(live: usize, reclaimed: usize) -> usize {
    let visited = live.saturating_add(reclaimed);
    let (growth_divisor, maximum_growth) = if reclaimed <= visited.saturating_div(20) {
        (4, MAXIMUM_LOW_YIELD_COLLECTION_GROWTH)
    } else if reclaimed <= visited.saturating_div(5) {
        (10, MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH)
    } else {
        (40, MAXIMUM_HIGH_YIELD_COLLECTION_GROWTH)
    };
    live.saturating_div(growth_divisor)
        .max(MINIMUM_HEAP_COLLECTION_GROWTH)
        .min(maximum_growth)
}

fn extend_heap_root_ids<'a>(
    datum_roots: &mut Vec<DatumId>,
    list_roots: &mut Vec<ListId>,
    values: impl IntoIterator<Item = &'a Value>,
) {
    for value in values {
        extend_heap_root_value(datum_roots, list_roots, value);
    }
}

fn extend_heap_root_value(
    datum_roots: &mut Vec<DatumId>,
    list_roots: &mut Vec<ListId>,
    value: &Value,
) {
    match value {
        Value::Datum(datum) => datum_roots.push(*datum),
        Value::List(list) => list_roots.push(*list),
        Value::ModifiedTypePath(path) => {
            for (_, value) in path.overrides() {
                extend_heap_root_value(datum_roots, list_roots, value);
            }
        }
        Value::Null | Value::Number(_) | Value::Text(_) | Value::File(_) | Value::TypePath(_) => {}
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CallFrame {
    pub(crate) procedure: ProcedureId,
    pub(crate) instruction: usize,
    // Most DM procedures use only a handful of locals and operand slots. Keep
    // those values inside the frame so millions of short startup calls do not
    // each pay for separate locals/stack heap allocations.
    pub(crate) locals: SmallVec<[Value; 8]>,
    pub(crate) stack: SmallVec<[Value; 8]>,
    pub(crate) result: Value,
    pub(crate) src: Value,
    pub(crate) usr: Value,
    // Retain all supplied values for the future DM `args` list, including
    // extras beyond the declared parameter slots.
    // The implicit DM `args` vector is needed for forwarding/default writes even
    // when its list identity is never materialized. Atom initialization calls
    // overwhelmingly supply only a few values, so keep that vector inline too.
    pub(crate) arguments: SmallVec<[Value; 8]>,
    // Materialized view of the special live `args` object. Parameter writes
    // keep its declared positions synchronized with `locals`.
    pub(crate) args_list: Option<ListId>,
    // Cache this per frame: StoreLocal is a dominant startup opcode and must
    // not rescan parameter_names for every ordinary local assignment.
    pub(crate) declared_argument_count: usize,
    pub(crate) supplied_parameters: SmallVec<[bool; 8]>,
    // Rare argument-name, exception, diagnostic, and shuttle continuation
    // state lives outside the hot frame header.
    pub(crate) cold: Option<Box<CallFrameCold>>,
    // A waitfor=FALSE boundary detaches from its caller only once. Later
    // sleeps in the already-detached continuation yield normally.
    pub(crate) detached_waitfor: bool,
    pub(crate) static_locals: SmallVec<[u16; 2]>,
    pub(crate) atoms_profile_entry_counted: bool,
    pub(crate) atoms_profile_root: bool,
    pub(crate) tgm_profile_root: bool,
}

/// Stable, native-width-independent identity for one suspended VM continuation.
///
/// The identity is serialized as an explicit `u64`; it never contains a host
/// pointer or depends on the 32-bit layout of BYOND's native runtime.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VmContinuationId(pub(crate) u64);

impl VmContinuationId {
    /// Returns the portable integer representation used by snapshots and diagnostics.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OwnedContinuation {
    pub(crate) id: VmContinuationId,
    pub(crate) frames: Vec<CallFrame>,
}

impl OwnedContinuation {
    pub(crate) fn new(id: VmContinuationId, frames: Vec<CallFrame>) -> Self {
        debug_assert!(!frames.is_empty());
        Self { id, frames }
    }

    pub(crate) fn into_frames(self) -> Vec<CallFrame> {
        self.frames
    }
}

impl std::ops::Deref for OwnedContinuation {
    type Target = [CallFrame];

    fn deref(&self) -> &Self::Target {
        &self.frames
    }
}

impl<'a> IntoIterator for &'a OwnedContinuation {
    type Item = &'a CallFrame;
    type IntoIter = std::slice::Iter<'a, CallFrame>;

    fn into_iter(self) -> Self::IntoIter {
        self.frames.iter()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PackedNumericState {
    pub(crate) locals: SmallVec<[PackedValue; 8]>,
    pub(crate) stack: SmallVec<[PackedValue; 8]>,
    pub(crate) result: PackedValue,
}

impl PackedNumericState {
    pub(crate) fn from_rich(frame: &CallFrame) -> Option<Self> {
        fn numeric(value: &Value) -> Option<PackedValue> {
            let packed = PackedValue::try_from_value(value)?;
            packed.as_number_or_null()?;
            Some(packed)
        }
        Some(Self {
            locals: frame.locals.iter().map(numeric).collect::<Option<_>>()?,
            stack: frame.stack.iter().map(numeric).collect::<Option<_>>()?,
            result: numeric(&frame.result)?,
        })
    }

    pub(crate) fn materialize(self, frame: &mut CallFrame) {
        frame.locals.clear();
        frame
            .locals
            .extend(self.locals.into_iter().map(PackedValue::into_value));
        frame.stack.clear();
        frame
            .stack
            .extend(self.stack.into_iter().map(PackedValue::into_value));
        frame.result = self.result.into_value();
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CallFrameCold {
    pub(crate) pending_argument_names: Option<Vec<Option<String>>>,
    // Source lists consumed by arglist expansion. Re-entrant engine execution
    // cannot always see the outer caller's frames, so transfer these roots to
    // the callee until it returns.
    pub(crate) pending_argument_roots: SmallVec<[Value; 2]>,
    pub(crate) retained_call_roots: SmallVec<[Value; 2]>,
    pub(crate) exception_handlers: Vec<ExceptionHandler>,
    pub(crate) shuttle_trace_target: Option<DatumId>,
    pub(crate) shuttle_trace_post_return: Option<ShuttleTracePostReturn>,
    pub(crate) boot_trace_started: Option<Instant>,
    pub(crate) boot_trace_heap: Option<(usize, usize, usize)>,
    pub(crate) boot_trace_step: u64,
    // Engine constructors return the allocated value rather than New()'s
    // result. This is rare outside constructor continuations.
    pub(crate) caller_result_override: Option<Value>,
    // Connection construction may defer Login until the complete New chain.
    pub(crate) engine_post_return: Option<Box<CallFrame>>,
    // Native numeric state exists only while a guarded trace is suspended.
    pub(crate) numeric_jit_state: Option<NumericExecutionState>,
    // Null/number-only interpreter state stays packed across budget yields.
    // Pointer-bearing values side-exit before entering this domain, so the
    // ordinary rich frame remains the complete GC root set.
    pub(crate) packed_numeric_state: Option<PackedNumericState>,
    pub(crate) tgm_load: Option<TgmLoadContinuation>,
    pub(crate) ruin_scan: Option<RuinCandidateScan>,
    pub(crate) tgm_route_trace_mask: u8,
}

#[derive(Clone, Debug)]
pub(crate) struct RuinCandidateScan {
    pub(crate) low: (i32, i32, i32),
    pub(crate) next: (i32, i32, i32),
    pub(crate) high: (i32, i32, i32),
    pub(crate) empty: bool,
    pub(crate) turfs: Vec<DatumId>,
    pub(crate) areas: Vec<Value>,
    pub(crate) validating: bool,
    pub(crate) validate_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TgmLoadPhase {
    Commit,
    AwaitCoordinate,
    Tick,
}

#[derive(Clone, Debug)]
pub(crate) struct TgmLoadContinuation {
    pub(crate) plan: Arc<tgm_planner::Plan>,
    pub(crate) cursor: tgm_planner::CommitCursor,
    pub(crate) phase: TgmLoadPhase,
    pub(crate) model_cache: Value,
    pub(crate) models: BTreeMap<Arc<str>, Value>,
    pub(crate) bounds: Value,
    pub(crate) coordinate_target: Option<(TypePath, ProcedureId)>,
}

impl TgmLoadContinuation {
    pub(crate) fn roots(&self) -> impl Iterator<Item = &Value> {
        std::iter::once(&self.model_cache)
            .chain(std::iter::once(&self.bounds))
            .chain(self.models.values())
    }
}

impl CallFrameCold {
    pub(crate) fn is_empty(&self) -> bool {
        self.pending_argument_names.is_none()
            && self.pending_argument_roots.is_empty()
            && self.retained_call_roots.is_empty()
            && self.exception_handlers.is_empty()
            && self.shuttle_trace_target.is_none()
            && self.shuttle_trace_post_return.is_none()
            && self.boot_trace_started.is_none()
            && self.boot_trace_heap.is_none()
            && self.boot_trace_step == 0
            && self.caller_result_override.is_none()
            && self.engine_post_return.is_none()
            && self.numeric_jit_state.is_none()
            && self.packed_numeric_state.is_none()
            && self.tgm_load.is_none()
            && self.ruin_scan.is_none()
            && self.tgm_route_trace_mask == 0
    }
}

impl CallFrame {
    pub(crate) fn prune_empty_cold(&mut self) {
        if self.cold.as_deref().is_some_and(CallFrameCold::is_empty) {
            self.cold = None;
        }
    }

    pub(crate) fn cold(&self) -> Option<&CallFrameCold> {
        self.cold.as_deref()
    }

    pub(crate) fn cold_mut(&mut self) -> &mut CallFrameCold {
        self.cold
            .get_or_insert_with(|| Box::new(CallFrameCold::default()))
    }

    pub(crate) fn pending_argument_names(&self) -> Option<&Vec<Option<String>>> {
        self.cold()
            .and_then(|cold| cold.pending_argument_names.as_ref())
    }

    pub(crate) fn set_pending_argument_names(&mut self, names: Vec<Option<String>>) {
        self.cold_mut().pending_argument_names = Some(names);
    }

    pub(crate) fn take_pending_argument_names(&mut self) -> Option<Vec<Option<String>>> {
        self.cold
            .as_deref_mut()
            .and_then(|cold| cold.pending_argument_names.take())
    }

    pub(crate) fn clear_pending_argument_names(&mut self) {
        if let Some(cold) = self.cold.as_deref_mut() {
            cold.pending_argument_names = None;
        }
    }

    pub(crate) fn pending_argument_roots(&self) -> &[Value] {
        self.cold()
            .map_or(&[], |cold| cold.pending_argument_roots.as_slice())
    }

    pub(crate) fn set_pending_argument_roots(&mut self, roots: SmallVec<[Value; 2]>) {
        if !roots.is_empty() || self.cold.is_some() {
            self.cold_mut().pending_argument_roots = roots;
        }
    }

    pub(crate) fn take_pending_argument_roots(&mut self) -> SmallVec<[Value; 2]> {
        self.cold.as_deref_mut().map_or_else(SmallVec::new, |cold| {
            std::mem::take(&mut cold.pending_argument_roots)
        })
    }

    pub(crate) fn clear_pending_argument_roots(&mut self) {
        if let Some(cold) = self.cold.as_deref_mut() {
            cold.pending_argument_roots.clear();
        }
    }

    pub(crate) fn retained_call_roots(&self) -> &[Value] {
        self.cold()
            .map_or(&[], |cold| cold.retained_call_roots.as_slice())
    }

    pub(crate) fn set_retained_call_roots(&mut self, roots: SmallVec<[Value; 2]>) {
        if !roots.is_empty() || self.cold.is_some() {
            self.cold_mut().retained_call_roots = roots;
        }
    }

    pub(crate) fn exception_handlers(&self) -> &[ExceptionHandler] {
        self.cold()
            .map_or(&[], |cold| cold.exception_handlers.as_slice())
    }

    pub(crate) fn exception_handlers_mut(&mut self) -> &mut Vec<ExceptionHandler> {
        &mut self.cold_mut().exception_handlers
    }

    pub(crate) fn shuttle_trace_target(&self) -> Option<DatumId> {
        self.cold().and_then(|cold| cold.shuttle_trace_target)
    }

    pub(crate) fn set_shuttle_trace_target(&mut self, target: Option<DatumId>) {
        if target.is_some() || self.cold.is_some() {
            self.cold_mut().shuttle_trace_target = target;
        }
    }

    pub(crate) fn take_shuttle_trace_post_return(&mut self) -> Option<ShuttleTracePostReturn> {
        self.cold
            .as_deref_mut()
            .and_then(|cold| cold.shuttle_trace_post_return.take())
    }

    pub(crate) fn set_shuttle_trace_post_return(&mut self, value: Option<ShuttleTracePostReturn>) {
        if value.is_some() || self.cold.is_some() {
            self.cold_mut().shuttle_trace_post_return = value;
        }
    }

    pub(crate) fn shuttle_trace_post_return(&self) -> Option<&ShuttleTracePostReturn> {
        self.cold()
            .and_then(|cold| cold.shuttle_trace_post_return.as_ref())
    }

    pub(crate) fn caller_result_override(&self) -> Option<&Value> {
        self.cold()
            .and_then(|cold| cold.caller_result_override.as_ref())
    }

    pub(crate) fn set_caller_result_override(&mut self, value: Option<Value>) {
        if value.is_some() || self.cold.is_some() {
            self.cold_mut().caller_result_override = value;
            self.prune_empty_cold();
        }
    }

    pub(crate) fn engine_post_return(&self) -> Option<&CallFrame> {
        self.cold()
            .and_then(|cold| cold.engine_post_return.as_deref())
    }

    pub(crate) fn set_engine_post_return(&mut self, frame: Option<Box<CallFrame>>) {
        if frame.is_some() || self.cold.is_some() {
            self.cold_mut().engine_post_return = frame;
        }
    }

    pub(crate) fn take_engine_post_return(&mut self) -> Option<Box<CallFrame>> {
        let frame = self
            .cold
            .as_deref_mut()
            .and_then(|cold| cold.engine_post_return.take());
        self.prune_empty_cold();
        frame
    }

    pub(crate) fn numeric_jit_state(&self) -> Option<&NumericExecutionState> {
        self.cold().and_then(|cold| cold.numeric_jit_state.as_ref())
    }

    pub(crate) fn numeric_jit_state_mut(&mut self) -> Option<&mut NumericExecutionState> {
        self.cold
            .as_deref_mut()
            .and_then(|cold| cold.numeric_jit_state.as_mut())
    }

    pub(crate) fn set_numeric_jit_state(&mut self, state: Option<NumericExecutionState>) {
        if state.is_some() || self.cold.is_some() {
            self.cold_mut().numeric_jit_state = state;
            self.prune_empty_cold();
        }
    }

    pub(crate) fn take_packed_numeric_state(&mut self) -> Option<PackedNumericState> {
        self.cold
            .as_deref_mut()
            .and_then(|cold| cold.packed_numeric_state.take())
    }

    pub(crate) fn set_packed_numeric_state(&mut self, state: Option<PackedNumericState>) {
        if state.is_some() || self.cold.is_some() {
            self.cold_mut().packed_numeric_state = state;
        }
        self.prune_empty_cold();
    }
}

pub(crate) enum FrameRunOutcome {
    Complete(Value),
    Yielded { frames: Vec<CallFrame>, delay: f32 },
    Prompted { id: u64, prompt: PendingLocalPrompt },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StepBudgetBehavior {
    Error,
    YieldScheduledContinuation,
}

fn materialize_callee_chain(
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

fn account_scheduler_tick_usage(state: &mut ExecutionState) {
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

pub(crate) fn make_frame(
    procedure: ProcedureId,
    program: &Program,
    arguments: &[Value],
    context: &ExecutionContext,
) -> CallFrame {
    make_frame_inline(
        procedure,
        program,
        arguments.iter().cloned().collect(),
        context,
    )
}

pub(crate) fn make_frame_owned(
    procedure: ProcedureId,
    program: &Program,
    arguments: impl Into<SmallVec<[Value; 8]>>,
    context: &ExecutionContext,
) -> CallFrame {
    make_frame_inline(procedure, program, arguments.into(), context)
}

fn make_frame_inline(
    procedure: ProcedureId,
    program: &Program,
    mut arguments: SmallVec<[Value; 8]>,
    context: &ExecutionContext,
) -> CallFrame {
    let declared_argument_count = declared_argument_count(program);
    let mut locals = SmallVec::<[Value; 8]>::from_elem(Value::Null, program.local_count);
    let bound_count = arguments
        .len()
        .min(program.parameter_count)
        .min(locals.len());
    locals[..bound_count].clone_from_slice(&arguments[..bound_count]);
    // BYOND's implicit `args` list has one entry for every declared
    // parameter even when the caller omitted it, padded with null, and then
    // retains any extra supplied arguments. This is observable in atom/New:
    // `new /obj` supplies no explicit location, but `/atom/New(loc, ...)`
    // can still assign `args[1]` before forwarding it to Initialize().
    let supplied_argument_count = arguments.len();
    arguments.resize(arguments.len().max(declared_argument_count), Value::Null);
    CallFrame {
        procedure,
        instruction: 0,
        locals,
        stack: SmallVec::new(),
        result: Value::Null,
        src: context.src.clone(),
        usr: context.usr.clone(),
        arguments,
        args_list: None,
        declared_argument_count,
        supplied_parameters: (0..program.parameter_count)
            .map(|index| index < supplied_argument_count)
            .collect(),
        cold: None,
        detached_waitfor: false,
        static_locals: SmallVec::new(),
        atoms_profile_entry_counted: false,
        atoms_profile_root: false,
        tgm_profile_root: false,
    }
}

fn make_frame_named(
    procedure: ProcedureId,
    program: &Program,
    arguments: &[Value],
    argument_names: &[Option<String>],
    context: &ExecutionContext,
) -> CallFrame {
    if argument_names.iter().all(Option::is_none) {
        return make_frame(procedure, program, arguments, context);
    }
    let mut positioned = vec![Value::Null; program.parameter_count];
    let mut extras = Vec::new();
    let mut supplied = vec![false; program.parameter_count];
    let mut next_positional = 0usize;
    for (index, value) in arguments.iter().enumerate() {
        let slot = argument_names
            .get(index)
            .and_then(Option::as_deref)
            .and_then(|name| {
                program
                    .parameter_names
                    .iter()
                    .position(|parameter| parameter == name)
            })
            .unwrap_or_else(|| {
                let slot = next_positional;
                next_positional += 1;
                slot
            });
        if slot < positioned.len() {
            positioned[slot] = value.clone();
            supplied[slot] = true;
        } else {
            extras.push(value.clone());
        }
    }
    let mut frame = make_frame(procedure, program, &positioned, context);
    frame.arguments = positioned[..declared_argument_count(program)]
        .iter()
        .cloned()
        .collect();
    frame.arguments.extend(extras);
    frame.supplied_parameters = supplied.into();
    frame
}

pub(crate) fn declared_argument_count(program: &Program) -> usize {
    // An unnamed trailing `...` reserves a compiler local slot but is not a
    // formal argument. BYOND's `args` pads omitted named parameters only; this
    // is why callback.New(thing, proc, ...) sees length(args) == 2 when no
    // captured callback arguments were supplied.
    program
        .parameter_names
        .iter()
        .rposition(|name| !name.is_empty())
        .map_or(0, |index| index + 1)
}

pub(crate) fn frame_context(frame: &CallFrame) -> ExecutionContext {
    ExecutionContext::new(frame.src.clone(), frame.usr.clone())
}

fn forwarded_frame_arguments(frame: &CallFrame, program: &Program) -> SmallVec<[Value; 8]> {
    // An argumentless self/parent call forwards the live parameter variables,
    // not the original pre-default call vector. Defaults and assignments have
    // already updated the parameter locals by this point. Preserve any extra
    // variadic values after the declared parameter slots.
    let mut arguments = frame.arguments.clone();
    for (index, value) in frame
        .locals
        .iter()
        .take(declared_argument_count(program))
        .enumerate()
    {
        arguments[index] = value.clone();
    }
    arguments
}

fn synchronize_frame_argument_write(
    frame: &mut CallFrame,
    program: &Program,
    key: &Value,
    value: Value,
) -> Result<(), String> {
    let index = value_to_list_index(key)?;
    if index == 0 || index > frame.arguments.len() {
        return Err(format!(
            "DM args position {index} exceeds length {}",
            frame.arguments.len(),
        ));
    }
    let slot = index - 1;
    frame.arguments[slot] = value.clone();
    if slot < declared_argument_count(program)
        && let Some(local) = frame.locals.get_mut(slot)
    {
        *local = value;
    }
    Ok(())
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

const DM_BIT_MASK: u32 = (1 << 24) - 1;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn dm_u24(value: f32) -> u32 {
    (value.trunc() as i64 as u32) & DM_BIT_MASK
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn bitwise_binary(
    left: f32,
    right: f32,
    operation: impl FnOnce(u32, u32) -> u32,
) -> f32 {
    (operation(dm_u24(left), dm_u24(right)) & DM_BIT_MASK) as f32
}

#[allow(clippy::cast_precision_loss)]
fn bitwise_not(value: f32) -> f32 {
    ((!dm_u24(value)) & DM_BIT_MASK) as f32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn bitwise_shift(left: f32, right: f32, operation: impl FnOnce(u32, u32) -> u32) -> f32 {
    let count = right.trunc().max(0.0) as u32;
    if count >= 24 {
        return 0.0;
    }
    (operation(dm_u24(left), count) & DM_BIT_MASK) as f32
}

#[cfg(test)]
#[test]
#[ignore = "explicit call-frame layout measurement"]
fn call_frame_layout_measurement() {
    eprintln!(
        "call-frame-layout value_bytes={} frame_bytes={} rare_inline_bytes_avoided={}",
        std::mem::size_of::<Value>(),
        std::mem::size_of::<CallFrame>(),
        std::mem::size_of::<Option<Value>>()
            + std::mem::size_of::<Option<Box<CallFrame>>>()
            + std::mem::size_of::<Option<NumericExecutionState>>(),
    );
}

#[cfg(test)]
#[test]
fn call_frame_hot_header_stays_within_one_kibibyte() {
    assert!(
        std::mem::size_of::<CallFrame>() <= 1_024,
        "cold state leaked back into the hot call-frame header"
    );
}

#[cfg(all(test, target_pointer_width = "64"))]
#[test]
fn call_frame_cold_split_reduces_the_hot_header_to_768_bytes() {
    assert_eq!(std::mem::size_of::<CallFrame>(), 768);
    let avoided = std::mem::size_of::<Option<Value>>()
        + std::mem::size_of::<Option<Box<CallFrame>>>()
        + std::mem::size_of::<Option<NumericExecutionState>>();
    assert!(
        avoided >= 200,
        "rare state no longer provides a useful split"
    );
}

#[cfg(test)]
#[test]
fn rare_frame_state_allocates_cold_storage_only_while_live() {
    let program = Program {
        wait_for: true,
        parameter_count: 0,
        parameter_names: Vec::new(),
        verb_parameter_types: Vec::new(),
        verb_name: None,
        local_count: 0,
        source_spans: vec![SourceSpan::new(0, 1)],
        instructions: vec![Instruction::Return],
    };
    let mut frame = make_frame(ProcedureId(0), &program, &[], &ExecutionContext::default());
    assert!(frame.cold.is_none());
    frame.set_caller_result_override(Some(Value::number(7.0)));
    assert_eq!(frame.caller_result_override(), Some(&Value::number(7.0)));
    assert!(frame.cold.is_some());
    frame.set_caller_result_override(None);
    assert!(frame.cold.is_none());
}

pub(crate) trait InlineValueStackExt {
    fn split_off(&mut self, at: usize) -> Vec<Value>;
}

impl InlineValueStackExt for SmallVec<[Value; 8]> {
    fn split_off(&mut self, at: usize) -> Vec<Value> {
        self.drain(at..).collect()
    }
}
pub(crate) fn preserve_reentrant_frame_roots(
    state: &mut ExecutionState,
    frames: &[CallFrame],
) -> usize {
    fn preserve_frame(state: &mut ExecutionState, frame: &CallFrame) {
        state.host_value_roots.extend(frame.locals.iter().cloned());
        state.host_value_roots.extend(frame.stack.iter().cloned());
        state.host_value_roots.push(frame.result.clone());
        state.host_value_roots.push(frame.src.clone());
        state.host_value_roots.push(frame.usr.clone());
        state
            .host_value_roots
            .extend(frame.arguments.iter().cloned());
        state
            .host_value_roots
            .extend(frame.pending_argument_roots().iter().cloned());
        state
            .host_value_roots
            .extend(frame.retained_call_roots().iter().cloned());
        if let Some(tgm) = frame.cold().and_then(|cold| cold.tgm_load.as_ref()) {
            state.host_value_roots.extend(tgm.roots().cloned());
        }
        if let Some(scan) = frame.cold().and_then(|cold| cold.ruin_scan.as_ref()) {
            state
                .host_value_roots
                .extend(scan.turfs.iter().copied().map(Value::Datum));
            state.host_value_roots.extend(scan.areas.iter().cloned());
        }
        if let Some(list) = frame.args_list {
            state.host_value_roots.push(Value::List(list));
        }
        if let Some(value) = frame.caller_result_override() {
            state.host_value_roots.push(value.clone());
        }
        if let Some(frame) = frame.engine_post_return() {
            preserve_frame(state, frame);
        }
    }

    let original_len = state.host_value_roots.len();
    for frame in frames {
        preserve_frame(state, frame);
    }
    original_len
}
/// Allocation-neutral scheduler metrics for continuation-layout benchmarks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ContinuationMetrics {
    /// Suspended continuations in scheduled and currently due queues.
    pub continuations: usize,
    /// Total call frames retained by those continuations.
    pub frames: usize,
    /// Frames carrying allocated cold exception/debug/argument state.
    pub cold_frames: usize,
    /// Rich values retained in locals, operand stacks, and argument vectors.
    pub retained_values: usize,
    /// Current native hot-frame header size, useful as a migration baseline.
    pub frame_header_bytes: usize,
    /// Header bytes avoided by moving rare state behind the cold allocation.
    pub rare_inline_bytes_avoided: usize,
}

/// Runtime counters for invalidation-safe static-field slot quickening.
///
/// The historical type name is retained for API compatibility. Counters now
/// include ordinary `datum.field` bytecode as well as statically declared
/// reads; dynamic `datum.vars[name]` accesses are not included.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeclaredFieldQuickeningMetrics {
    /// Reads served by a validated dense slot.
    pub hits: u64,
    /// Reads that resolved and installed or refreshed a slot.
    pub misses: u64,
    /// Cached slots rejected after a datum layout mutation.
    pub invalidations: u64,
}
