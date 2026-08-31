//! The interpreter execution state: the heap, globals, scheduler, local-client sessions, and type metadata.
//!
//! Module layout: this crate splits `execution` into `state`, `frame`, `scheduler`, `run`,
//! and `support` so the deterministic engine stays navigable by concern.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::thread::ThreadId;
use std::time::Instant;

use crate::builtins;
use crate::bytecode::{InstanceInitializer, Module, ProcedureId};
use crate::{
    AtomsProfile, ClientState, DmmMeasurement, GlobalStore,
    MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES, MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE,
    NativeWalk, ParsedDmm, PendingPromptContinuation, QuiescentHeapCompaction, SavefileState,
    ScheduledSpawn, TgmProfile, boot_dashboard_enabled, boot_trace_enabled, datum_field_or_initial,
    datum_shared_storage, engine_builtin_initial_fields, engine_builtin_initial_value,
    engine_root_initial_value, initialize_existing_datum, lazy_atom_list_field,
};
use dm_jit::NumericExecutionState;
use dm_value::{DatumId, FieldName, ListId, TypePath, Value, ValueHeap};

use crate::execution::frame::CallFrame;
use crate::execution::support::ContinuationMetrics;
use crate::execution::support::DeclaredFieldQuickeningMetrics;

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
        self.scheduled_spawns.len() + self.client.pending_prompt_count()
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
        datum_roots.extend(self.client.session_datums());
        datum_roots.extend(self.client.mob_datums());
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
        for prompt in self.client.pending_prompts() {
            datum_roots.push(prompt.1.client);
            extend_heap_root_ids(&mut datum_roots, &mut list_roots, &prompt.1.choices);
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
                    current.caller_result_override(),
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
        for (_, prompt) in self.client.pending_prompts() {
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
    if !fields.is_empty() {
        format!(" map=[{}]", fields.join(","))
    } else {
        Default::default()
    }
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
