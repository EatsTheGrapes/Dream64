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
use crate::value_ops::ExecutionContext;
use crate::{
    AtomsProfile, ClientState, DmmMeasurement, GlobalStore, HeapReference, LocalClientAppearance,
    LocalClientError, LocalClientMapSnapshot, LocalClientMapTile, LocalClientPromptKind,
    LocalClientPromptResponse, LocalClientScreenAppearance, LocalClientState, LocalClientUiEvent,
    LocalMovementDirection, LocalScreenPointerEvent, MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES,
    MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE, NativeWalk, ParsedDmm,
    PendingPromptContinuation, PendingVerbInvocation, QuiescentHeapCompaction, SavefileState,
    ScheduledSpawn, TgmProfile, VerbParameterType, allocate_initialized_datum, assign_datum_field,
    boot_dashboard_enabled, boot_trace_enabled, datum_field_or_initial, datum_shared_storage,
    dynamic_call_target_named, engine_builtin_initial_fields, engine_builtin_initial_value,
    engine_root_initial_value, initialize_existing_datum, instance_initializer_plan,
    is_atom_type_path, lazy_atom_list_field, parse_heap_reference, queue_next_verb_prompt,
};
use dm_dmf::{ClientSession, ControlTree, DiagnosticSeverity, UiEvent, parse as parse_dmf};
use dm_jit::NumericExecutionState;
use dm_value::{DatumId, FieldName, ListId, TypePath, Value, ValueHeap};

use crate::execution::frame::CallFrame;
use crate::execution::frame::make_frame;
use crate::execution::scheduler::schedule_frames;
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
        self.client.install_session(client, tree);
    }

    /// Sets the parsed skin cloned into subsequently connected local clients.
    pub fn set_local_client_skin(&mut self, tree: ControlTree) {
        self.client.set_skin(tree);
    }

    /// Drains authoritative UI operations in exact DM execution order.
    #[must_use]
    pub fn take_local_client_outbound_events(
        &mut self,
        client: DatumId,
    ) -> Vec<LocalClientUiEvent> {
        self.client.take_outbound_events().remove(&client).unwrap_or_default()
    }

    pub(crate) fn emit_local_client_ui_event(
        &mut self,
        client: DatumId,
        event: LocalClientUiEvent,
    ) {
        self.client.emit_ui_event(client, event);
    }

    /// Returns the number of DM continuations waiting for native prompt input.
    #[must_use]
    pub fn pending_local_prompt_count(&self) -> usize {
        self.client.pending_prompt_count()
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
            .client
            .take_prompt(id)
            .ok_or_else(|| format!("unknown local prompt {id}"))?;
        if prompt.client != client {
            self.client.register_prompt(id, prompt);
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
                self.client.register_prompt(id, prompt);
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
                    self.client.register_prompt(id, prompt.clone());
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
                    self.client.register_prompt(id, prompt.clone());
                    format!("local prompt {id} choice {index} is out of range")
                })?
            }
            _ => {
                self.client.register_prompt(id, prompt);
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
        self.client.client_sessions.get(&client)
    }

    /// Returns the mutable UI session associated with a connected client datum.
    pub fn client_session_mut(&mut self, client: DatumId) -> Option<&mut ClientSession> {
        self.client.client_sessions.get_mut(&client)
    }

    /// Enables or disables modal prompt suspension for a window-attached
    /// client. Skin-only preflight clients remain non-interactive so startup
    /// probes cannot deadlock waiting for a UI response that has no consumer.
    pub fn set_local_client_interactive(
        &mut self,
        client: DatumId,
        interactive: bool,
    ) -> Result<(), String> {
        if !self.client.has_session(client) {
            return Err("local client has no installed UI session".to_owned());
        }
        self.client.set_interactive(client, interactive);
        Ok(())
    }

    /// Drains local UI events emitted by one connected client.
    #[must_use]
    pub fn take_client_events(&mut self, client: DatumId) -> Vec<UiEvent> {
        self.client
            .client_sessions
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
        self.client.attach_mob(client, mob);
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
        self.client.attach_mob(client, mob);
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
        if self.client.local_client_skin.is_none()
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
                self.client.set_skin(ControlTree::from_document(&document));
            }
        }
        let attached = self.create_pending_local_client()?;
        self.populate_local_verb_inventory(module, attached.client)?;
        self.populate_local_verb_inventory(module, attached.mob)?;
        let sequence = self.client.next_guest_id();
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
            self.client.local_client_skin.clone().unwrap_or_default(),
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
            .client
            .attached_mob(client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        self.heap.datum(mob).map_err(|error| error.to_string())?;
        self.client.queue_command(client, direction);
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
        let mob = self
            .client
            .attached_mob(client)
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
                    Value::Text(reference) => match parse_heap_reference(reference) {
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
        let mob = self
            .client
            .attached_mob(client)
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
        let list = if let Ok(Value::List(list)) = self.heap.datum_field(datum, &verbs_field) {
            *list
        } else {
            let list = self.heap.allocate_list();
            self.heap
                .set_datum_field(datum, verbs_field, Value::List(list))
                .map_err(|error| error.to_string())?;
            list
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
        let mut commands = self.client.take_commands();
        commands.sort_by_key(|(sequence, _, _)| *sequence);
        let mut committed = Vec::with_capacity(commands.len());
        for (_, client, direction) in commands {
            let mob = self
                .client
                .attached_mob(client)
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
        let mob = self
            .client
            .attached_mob(client)
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
        let mob = self
            .client
            .attached_mob(client)
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
        let mob = self
            .client
            .attached_mob(client)
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
        let mob = self
            .client
            .attached_mob(client)
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

fn normalize_client_command_name(name: &str) -> String {
    name.chars()
        .filter(char::is_ascii_alphanumeric)
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
