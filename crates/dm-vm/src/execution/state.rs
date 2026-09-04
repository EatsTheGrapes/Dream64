//! The interpreter execution state: the heap, globals, scheduler, local-client sessions, and type metadata.
//!
//! Module layout: this crate splits `execution` into `state`, `frame`, `scheduler`, `run`,
//! and `support` so the deterministic engine stays navigable by concern.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::thread::ThreadId;
use std::time::Instant;

use smallvec::SmallVec;

use crate::builtins;
use crate::bytecode::{InstanceInitializer, Module, ProcedureId};
use crate::{
    AtomsProfile, ClientState, DmmMeasurement, GlobalStore, NativeWalk, ParsedDmm, SavefileState,
    ScheduledSpawn, TgmProfile, datum_field_or_initial, datum_shared_storage,
    initialize_existing_datum, lazy_atom_list_field,
};
use dm_value::{DatumId, FieldName, ListId, TypePath, Value, ValueHeap};

use crate::execution::support::DeclaredFieldQuickeningMetrics;

/// How a `(call site, receiver type)` field read resolves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FieldResolution {
    /// The field is materialized on the instance at this physical slot. Still
    /// revalidated against the live datum on every hit.
    Slot(u16),
    /// The field is not materialized on the instance and resolves through
    /// `effective_initial_value(type, field)` — i.e. it is a genuine
    /// instance-defaulted variable, not a shared/static var (those are absent
    /// from that catalog) and not a missing field (that returns `None`). The
    /// hit path re-derives the value live and re-checks that the field is
    /// still unmaterialized, so runtime mutation and catalog changes are
    /// always observed.
    InitialValue,
}

/// A polymorphic inline cache for one `obj.field` read site.
///
/// A static field-read instruction has a fixed field *name*, but how that name
/// resolves depends on the receiver's type. SS13 init is polymorphic at the
/// call site — `/atom/proc/Initialize`, the signal dispatchers,
/// `update_appearance` all run the same `LoadField` over turfs, objs, mobs and
/// areas — so a single cached resolution per site thrashes on every type
/// switch. This keeps one resolution *per receiver type* the site has seen.
/// Every hit revalidates against the live datum and current catalog, so the
/// type key only affects the hit rate, never correctness.
#[derive(Default)]
pub(crate) struct FieldSlotCache {
    /// `(receiver type, resolution)`, most-recently-used last. Inline capacity
    /// covers the overwhelming majority of call sites; a wider site spills to
    /// the heap and is capped at [`Self::MAX_ENTRIES`].
    entries: SmallVec<[(TypePath, FieldResolution); 4]>,
}

impl FieldSlotCache {
    /// Hard cap on distinct receiver types tracked per site. A site past this
    /// is effectively megamorphic; it keeps its most-recent types and lets the
    /// rest fall to the ordinary path (still better than the old thrash).
    pub(crate) const MAX_ENTRIES: usize = 16;

    /// The cached resolution for `receiver_type`, promoted to most-recently-used.
    pub(crate) fn resolution_for(&mut self, receiver_type: &TypePath) -> Option<FieldResolution> {
        let index = self
            .entries
            .iter()
            .position(|(cached, _)| cached == receiver_type)?;
        if index + 1 != self.entries.len() {
            let entry = self.entries.remove(index);
            self.entries.push(entry);
        }
        Some(self.entries.last().expect("just pushed").1)
    }

    /// Records `resolution` for `receiver_type`, evicting the least-recently-used
    /// entry once the cap is reached.
    pub(crate) fn remember(&mut self, receiver_type: TypePath, resolution: FieldResolution) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|(cached, _)| *cached == receiver_type)
        {
            existing.1 = resolution;
            return;
        }
        if self.entries.len() >= Self::MAX_ENTRIES {
            self.entries.remove(0);
        }
        self.entries.push((receiver_type, resolution));
    }

    /// Drops the entry for `receiver_type` after its resolution failed
    /// revalidation.
    pub(crate) fn forget(&mut self, receiver_type: &TypePath) {
        self.entries.retain(|(cached, _)| cached != receiver_type);
    }

    #[cfg(test)]
    pub(crate) fn tracked_type_count(&self) -> usize {
        self.entries.len()
    }
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
    pub(crate) client: ClientState,
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
    // Static field-read sites retain a dense datum slot per receiver type they
    // have observed. Every hit still validates the field name and current
    // layout, and engine-special fields never enter this path. See
    // [`FieldSlotCache`].
    pub(crate) field_slot_cache: HashMap<(u64, ProcedureId, u16), FieldSlotCache>,
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
    // Diagnostic: `(cache hits, cold resolves that walked the ancestry)` for
    // `effective_initial_value`. Reveals whether the field-read miss path's
    // cost is the walk itself or the surrounding machinery.
    pub(crate) effective_initial_value_hits: Cell<u64>,
    pub(crate) effective_initial_value_cold: Cell<u64>,
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
    /// GAGS config path -> (source template DMI, raw config JSON). The JSON is
    /// retained so `iconforge_gags` can composite the layer stacks natively.
    pub(crate) iconforge_gags_configs: BTreeMap<String, (PathBuf, String)>,
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
    /// Cumulative executed-collection count and the wall time they took. Cheap
    /// to maintain (a collection is infrequent) and always on, so a host can
    /// attribute boot wall time to GC without a trace build.
    pub(crate) list_gc_count: u64,
    pub(crate) list_gc_elapsed: std::time::Duration,
    /// Cumulative interpreted instruction steps executed by `run_frames`,
    /// flushed at each frame-run boundary. Always on; one add per boundary.
    pub(crate) total_executed_steps: u64,
    /// Per-procedure self-time step counts. Populated only while
    /// `DREAM64_PROFILE_PROC_STEPS` is set.
    pub(crate) proc_step_samples: HashMap<ProcedureId, u64>,
    /// Whole-boot instruction-category histogram. `Some` only while
    /// `DREAM64_PROFILE_INSTRUCTIONS` is set.
    pub(crate) instruction_profile: Option<Box<crate::StartupInstructionProfile>>,
    /// Consecutive near-zero-yield list collections. A run of these identifies a
    /// monotonic bulk-allocation phase and widens the next collection window.
    pub(crate) low_yield_collection_streak: u32,
    /// Live-identity ceiling that forces a collection even mid-bulk-init so
    /// committed memory stays bounded. Resolved once from the environment.
    pub(crate) heap_identity_ceiling: usize,
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
            client: ClientState::new(),
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
            field_slot_cache: HashMap::new(),
            declared_field_quickening: DeclaredFieldQuickeningMetrics::default(),
            initial_values: Arc::new(BTreeMap::new()),
            initial_value_datum_roots: Arc::default(),
            initial_value_list_roots: Arc::default(),
            effective_initial_value_cache: RefCell::new(HashMap::new()),
            effective_initial_value_cache_entries: Cell::new(0),
            effective_initial_value_hits: Cell::new(0),
            effective_initial_value_cold: Cell::new(0),
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
            list_gc_count: 0,
            list_gc_elapsed: std::time::Duration::ZERO,
            total_executed_steps: 0,
            proc_step_samples: HashMap::new(),
            instruction_profile: crate::instruction_profile_enabled()
                .then(|| Box::new(crate::StartupInstructionProfile::default())),
            low_yield_collection_streak: 0,
            heap_identity_ceiling: super::heap_gc::resolve_heap_identity_ceiling(),
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

    /// Cumulative heap-collection count and wall time since this state existed.
    #[must_use]
    pub fn list_gc_totals(&self) -> (u64, std::time::Duration) {
        (self.list_gc_count, self.list_gc_elapsed)
    }

    /// Cumulative interpreted instruction steps executed since this state existed.
    #[must_use]
    pub fn total_executed_steps(&self) -> u64 {
        self.total_executed_steps
    }

    /// `DREAM64_PROFILE_PROC_STEPS` accounting: the `limit` procedures with the
    /// most self-time steps, as `(path, steps)` descending. Empty unless set.
    #[must_use]
    pub fn proc_step_profile_top(&self, module: &Module, limit: usize) -> Vec<(String, u64)> {
        let mut rows: Vec<(String, u64)> = self
            .proc_step_samples
            .iter()
            .map(|(procedure, steps)| {
                (
                    module
                        .procedure_path(*procedure)
                        .unwrap_or("<missing>")
                        .to_owned(),
                    *steps,
                )
            })
            .collect();
        rows.sort_unstable_by(|left, right| right.1.cmp(&left.1));
        rows.truncate(limit);
        rows
    }

    /// Marks the boundary between startup initialization and steady-state
    /// execution for `DREAM64_PROFILE_INSTRUCTIONS`. The host calls this once,
    /// when it opens the lobby for connections.
    pub fn mark_profile_steady_state(&mut self) {
        if let Some(profile) = &mut self.instruction_profile {
            profile.mark_steady_state();
        }
    }

    /// `DREAM64_PROFILE_INSTRUCTIONS` histogram for the given phase
    /// (`false` = startup, `true` = steady-state), most expensive first. Empty
    /// unless the profiler is enabled.
    #[must_use]
    pub fn instruction_profile_lines(&self, phase_steady: bool) -> Vec<String> {
        self.instruction_profile
            .as_ref()
            .map_or_else(Vec::new, |profile| profile.lines(phase_steady))
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

    /// Returns the project root used for relative filesystem paths.
    #[must_use]
    pub fn project_root(&self) -> Option<&std::path::Path> {
        self.project_root.as_deref().map(PathBuf::as_path)
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
        self.scheduled_spawns.len() + self.client.pending_prompt_count()
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

    /// `(cache hits, cold ancestry walks)` for `effective_initial_value`.
    #[must_use]
    pub fn effective_initial_value_totals(&self) -> (u64, u64) {
        (
            self.effective_initial_value_hits.get(),
            self.effective_initial_value_cold.get(),
        )
    }

    /// The most receiver types any single field-read call site is tracking in
    /// the inline cache — for asserting megamorphic sites stay bounded.
    #[cfg(test)]
    pub(crate) fn field_slot_cache_widest_site(&self) -> usize {
        self.field_slot_cache
            .values()
            .map(FieldSlotCache::tracked_type_count)
            .max()
            .unwrap_or(0)
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
}
