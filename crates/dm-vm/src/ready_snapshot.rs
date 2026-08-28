use super::*;
use bincode::Options as _;
use dm_value::{HeapSnapshotValue, ValueHeapSnapshot};
use std::io::{self, Read, Write};

type DatumHandle = (u32, u32);
type ListHandle = (u32, u32);

const READY_WORLD_MAGIC: &[u8; 8] = b"D64READY";
const READY_WORLD_VERSION: u32 = 1;
const READY_WORLD_METADATA_LIMIT: u64 = 1024 * 1024 * 1024;
const RUNTIME_CATALOG_MAGIC: &[u8; 8] = b"D64RCAT\0";
const RUNTIME_CATALOG_VERSION: u32 = 1;

#[derive(serde::Deserialize, serde::Serialize)]
struct RuntimeCatalogSnapshot {
    type_paths: Vec<String>,
    type_parents: Vec<(String, Option<String>)>,
    initial_values: Vec<(String, Vec<(String, HeapSnapshotValue)>)>,
    shared_fields: Vec<(String, Vec<(String, String)>)>,
    instance_initializers: Vec<(String, Vec<RuntimeInitializerSnapshot>)>,
    initializer_module: Option<Vec<u8>>,
    project_root: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize)]
enum RuntimeInitializerSnapshot {
    Constant(String, HeapSnapshotValue),
    Program(String, u32),
}

/// Mutable, process-independent runtime state captured at the ready boundary.
///
/// Immutable type/procedure/default catalogs remain in the executable artifact
/// and are reused by the destination [`ExecutionState`]. Host connections,
/// wall-clock values, profilers, external jobs, and derived dispatch indexes
/// are deliberately excluded and rebound after restore.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ReadyWorldCoreSnapshot {
    heap: ValueHeapSnapshot,
    associative_lists: Vec<ListHandle>,
    reference_lists: Vec<ListHandle>,
    savefiles: Vec<(DatumHandle, String, Vec<(String, HeapSnapshotValue)>)>,
    savefile_entries: Vec<(DatumHandle, DatumHandle, String)>,
    global_vars_proxy: Option<ListHandle>,
    datum_vars_proxies: Vec<(ListHandle, DatumHandle)>,
    datum_vars_by_datum: Vec<(DatumHandle, ListHandle)>,
    initial_prototypes: Vec<(String, DatumHandle)>,
    globals: Vec<(String, HeapSnapshotValue)>,
    compact_default_datums: Vec<DatumHandle>,
    random_state: u64,
    scheduler_tick: u64,
    scheduler_sequence: u64,
    scheduled_spawns: Vec<ScheduledSpawnSnapshot>,
    native_walks: Vec<(DatumHandle, NativeWalkSnapshot)>,
    last_animation_target: Option<HeapSnapshotValue>,
    environment_overrides: Vec<(String, Option<HeapSnapshotValue>)>,
    external_timers: Vec<(String, u64)>,
    iconforge_jobs: Vec<(String, bool, String)>,
    iconforge_next_job: u64,
    iconforge_gags_configs: Vec<(String, String)>,
    sql_jobs: Vec<(String, bool, String)>,
    sql_next_job: u64,
    procedure_static_locals: Vec<(String, Vec<(u16, HeapSnapshotValue)>)>,
    default_world_area: Option<DatumHandle>,
}

impl ReadyWorldCoreSnapshot {
    /// Validates every persisted continuation against the executable artifact
    /// that will resume it.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic when a procedure identity, instruction, local
    /// layout, or exception target no longer matches the module. A cache-key
    /// collision or stale image therefore becomes a cache miss, not execution
    /// of an incompatible frame.
    pub fn validate_module(&self, module: &Module) -> Result<(), String> {
        for spawn in &self.scheduled_spawns {
            if spawn.frames.is_empty() {
                return Err("ready-world scheduler record has no call frames".into());
            }
            for frame in &spawn.frames {
                validate_frame(module, frame, 0)?;
            }
        }
        Ok(())
    }
}

impl ExecutionState {
    /// Writes immutable runtime catalogs without process pointers or OS handles.
    pub fn write_runtime_catalog_to(&self, writer: &mut impl Write) -> io::Result<()> {
        let snapshot = RuntimeCatalogSnapshot {
            type_paths: self.type_paths.iter().map(ToString::to_string).collect(),
            type_parents: self
                .type_parents
                .iter()
                .map(|(path, parent)| (path.to_string(), parent.as_ref().map(ToString::to_string)))
                .collect(),
            initial_values: self
                .initial_values
                .iter()
                .map(|(path, fields)| {
                    (
                        path.to_string(),
                        fields
                            .iter()
                            .map(|(field, value)| {
                                (field.as_str().to_owned(), HeapSnapshotValue::from(value))
                            })
                            .collect(),
                    )
                })
                .collect(),
            shared_fields: self
                .shared_fields
                .iter()
                .map(|(path, fields)| {
                    (
                        path.to_string(),
                        fields
                            .iter()
                            .map(|(field, shared)| {
                                (field.as_str().to_owned(), shared.as_str().to_owned())
                            })
                            .collect(),
                    )
                })
                .collect(),
            instance_initializers: self
                .instance_initializers
                .iter()
                .map(|(path, initializers)| {
                    (
                        path.to_string(),
                        initializers
                            .iter()
                            .map(|initializer| match initializer {
                                InstanceInitializer::Constant { field, value } => {
                                    RuntimeInitializerSnapshot::Constant(
                                        field.as_str().to_owned(),
                                        HeapSnapshotValue::from(value),
                                    )
                                }
                                InstanceInitializer::Program { field, entry } => {
                                    RuntimeInitializerSnapshot::Program(
                                        field.as_str().to_owned(),
                                        entry.0,
                                    )
                                }
                            })
                            .collect(),
                    )
                })
                .collect(),
            initializer_module: self
                .instance_initializer_module
                .as_deref()
                .map(Module::encode_portable)
                .transpose()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
            project_root: self
                .project_root()
                .map(|path| path.to_string_lossy().into_owned()),
        };
        writer.write_all(RUNTIME_CATALOG_MAGIC)?;
        writer.write_all(&RUNTIME_CATALOG_VERSION.to_le_bytes())?;
        ready_world_bincode()
            .serialize_into(writer, &snapshot)
            .map_err(bincode_io_error)
    }

    /// Restores immutable runtime catalogs written by [`Self::write_runtime_catalog_to`].
    pub fn restore_runtime_catalog_from(&mut self, reader: &mut impl Read) -> io::Result<()> {
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != RUNTIME_CATALOG_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "runtime catalog magic",
            ));
        }
        let mut version = [0; 4];
        reader.read_exact(&mut version)?;
        if u32::from_le_bytes(version) != RUNTIME_CATALOG_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "runtime catalog version",
            ));
        }
        let snapshot: RuntimeCatalogSnapshot = ready_world_bincode()
            .deserialize_from(reader)
            .map_err(bincode_io_error)?;
        let parse_path = |path: String| {
            TypePath::parse(&path)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        };
        let parse_field = |field: String| {
            FieldName::parse(&field)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        };
        self.set_type_paths(
            snapshot
                .type_paths
                .into_iter()
                .map(parse_path)
                .collect::<io::Result<Vec<_>>>()?,
        );
        self.set_type_parents(
            snapshot
                .type_parents
                .into_iter()
                .map(|(path, parent)| Ok((parse_path(path)?, parent.map(parse_path).transpose()?)))
                .collect::<io::Result<_>>()?,
        );
        self.set_initial_values(
            snapshot
                .initial_values
                .into_iter()
                .map(|(path, fields)| {
                    Ok((
                        parse_path(path)?,
                        fields
                            .into_iter()
                            .map(|(field, value)| {
                                Ok((
                                    parse_field(field)?,
                                    value.into_value().map_err(|error| {
                                        io::Error::new(io::ErrorKind::InvalidData, error)
                                    })?,
                                ))
                            })
                            .collect::<io::Result<_>>()?,
                    ))
                })
                .collect::<io::Result<_>>()?,
        );
        self.set_shared_fields(Arc::new(
            snapshot
                .shared_fields
                .into_iter()
                .map(|(path, fields)| {
                    Ok((
                        parse_path(path)?,
                        fields
                            .into_iter()
                            .map(|(field, shared)| Ok((parse_field(field)?, parse_field(shared)?)))
                            .collect::<io::Result<_>>()?,
                    ))
                })
                .collect::<io::Result<_>>()?,
        ));
        let initializers = snapshot
            .instance_initializers
            .into_iter()
            .map(|(path, initializers)| {
                Ok((
                    parse_path(path)?,
                    initializers
                        .into_iter()
                        .map(|initializer| match initializer {
                            RuntimeInitializerSnapshot::Constant(field, value) => {
                                Ok(InstanceInitializer::Constant {
                                    field: parse_field(field)?,
                                    value: value.into_value().map_err(|error| {
                                        io::Error::new(io::ErrorKind::InvalidData, error)
                                    })?,
                                })
                            }
                            RuntimeInitializerSnapshot::Program(field, entry) => {
                                Ok(InstanceInitializer::Program {
                                    field: parse_field(field)?,
                                    entry: ProcedureId(entry),
                                })
                            }
                        })
                        .collect::<io::Result<_>>()?,
                ))
            })
            .collect::<io::Result<_>>()?;
        let module = snapshot
            .initializer_module
            .map(|bytes| Module::decode_portable(&bytes))
            .transpose()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
            .map(Arc::new);
        self.set_instance_initializers(Arc::new(initializers), module);
        if let Some(root) = snapshot.project_root {
            self.set_project_root(PathBuf::from(root));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct ScheduledSpawnSnapshot {
    due_tick: u64,
    sequence: u64,
    frames: Vec<CallFrameSnapshot>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct CallFrameSnapshot {
    procedure: u32,
    instruction: usize,
    locals: Vec<HeapSnapshotValue>,
    stack: Vec<HeapSnapshotValue>,
    result: HeapSnapshotValue,
    src: HeapSnapshotValue,
    usr: HeapSnapshotValue,
    arguments: Vec<HeapSnapshotValue>,
    args_list: Option<ListHandle>,
    declared_argument_count: usize,
    supplied_parameters: Vec<bool>,
    pending_argument_names: Option<Vec<Option<String>>>,
    pending_argument_roots: Vec<HeapSnapshotValue>,
    retained_call_roots: Vec<HeapSnapshotValue>,
    exception_handlers: Vec<ExceptionHandlerSnapshot>,
    detached_waitfor: bool,
    caller_result_override: Option<HeapSnapshotValue>,
    engine_post_return: Option<Box<CallFrameSnapshot>>,
    static_locals: Vec<u16>,
    shuttle_trace_target: Option<DatumHandle>,
    shuttle_trace_post_return: Option<ShuttleTracePostReturnSnapshot>,
    numeric_jit_state: Option<NumericStateSnapshot>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct ExceptionHandlerSnapshot {
    start: usize,
    end: usize,
    catch: usize,
    local: Option<u16>,
    stack_depth: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
enum ShuttleTracePostReturnSnapshot {
    AtmosInit,
    NullifyNode { slot: Option<usize> },
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct NumericStateSnapshot {
    locals: Vec<f32>,
    stack: Vec<f32>,
    fields: Vec<f32>,
    dirty_fields: u64,
    action_bits: u64,
    instruction: u32,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct NativeWalkSnapshot {
    due_tick: u64,
    sequence: u64,
    lag: u64,
    kind: NativeWalkKindSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
enum NativeWalkKindSnapshot {
    Direction(i16),
    Random,
    Towards(DatumHandle),
    To { target: DatumHandle, minimum: f32 },
    Away { target: DatumHandle, maximum: f32 },
}

impl ExecutionState {
    /// Streams a complete mutable ready-world state image without cloning the
    /// live heap.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] unless called between scheduler
    /// slices before clients/prompts are attached. Destination and encoding
    /// failures are returned unchanged.
    pub fn write_ready_world_snapshot_to(&self, writer: &mut impl Write) -> io::Result<()> {
        if !self.scheduler_inflight.is_empty()
            || !self.client_sessions.is_empty()
            || !self.pending_local_prompts.is_empty()
            || !self.deleting_datums.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ready-world snapshot requires a scheduler boundary before client attachment",
            ));
        }
        writer.write_all(READY_WORLD_MAGIC)?;
        writer.write_all(&READY_WORLD_VERSION.to_le_bytes())?;
        self.heap.write_snapshot_to(writer)?;
        let metadata = self.capture_ready_world_metadata();
        ready_world_bincode()
            .serialize_into(writer, &metadata)
            .map_err(bincode_io_error)
    }

    /// Loads a complete ready-world image into this artifact-initialized state.
    ///
    /// The executable module is checked before any mutable state is committed.
    /// Immutable type/default/procedure catalogs already installed on `self`
    /// are retained and process-bound host state is rebound.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidData`] for a stale/corrupt image or an
    /// executable-continuation mismatch.
    pub fn restore_ready_world_snapshot_from(
        &mut self,
        reader: &mut impl Read,
        module: &Module,
    ) -> io::Result<()> {
        self.assert_owner_thread();
        let mut magic = [0_u8; READY_WORLD_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != READY_WORLD_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ready-world state magic does not match",
            ));
        }
        let mut version = [0_u8; 4];
        reader.read_exact(&mut version)?;
        let version = u32::from_le_bytes(version);
        if version != READY_WORLD_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported ready-world state version {version}; expected {READY_WORLD_VERSION}"
                ),
            ));
        }
        let heap = ValueHeap::read_snapshot_from(reader)?;
        let metadata: ReadyWorldCoreSnapshot = ready_world_bincode()
            .deserialize_from(reader)
            .map_err(bincode_io_error)?;
        if !metadata.heap.datums.is_empty()
            || !metadata.heap.datum_free.is_empty()
            || !metadata.heap.lists.is_empty()
            || !metadata.heap.list_free.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ready-world metadata contains a duplicate heap section",
            ));
        }
        metadata
            .validate_module(module)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.restore_ready_world_parts(metadata, heap)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    /// Captures the heap and mutable core tables needed by a ready-world image.
    ///
    /// Scheduler continuations are a separate section because they reference
    /// executable procedure identities and require module-key validation.
    #[must_use]
    pub fn capture_ready_world_core(&self) -> ReadyWorldCoreSnapshot {
        let mut snapshot = self.capture_ready_world_metadata();
        snapshot.heap = self.heap.snapshot();
        snapshot
    }

    fn capture_ready_world_metadata(&self) -> ReadyWorldCoreSnapshot {
        ReadyWorldCoreSnapshot {
            heap: empty_heap_snapshot(),
            associative_lists: sorted_list_handles(&self.associative_lists),
            reference_lists: sorted_list_handles(&self.reference_lists),
            savefiles: self
                .savefiles
                .iter()
                .map(|(datum, savefile)| {
                    let mut entries = savefile
                        .entries
                        .iter()
                        .map(|(name, value)| (name.clone(), HeapSnapshotValue::from(value)))
                        .collect::<Vec<_>>();
                    entries.sort_by(|left, right| left.0.cmp(&right.0));
                    (datum_handle(*datum), savefile.cd.clone(), entries)
                })
                .collect(),
            savefile_entries: self
                .savefile_entries
                .iter()
                .map(|(entry, (owner, name))| {
                    (datum_handle(*entry), datum_handle(*owner), name.clone())
                })
                .collect(),
            global_vars_proxy: self.global_vars_proxy.map(list_handle),
            datum_vars_proxies: sorted_handle_map(
                &self.datum_vars_proxies,
                list_handle,
                datum_handle,
            ),
            datum_vars_by_datum: sorted_handle_map(
                &self.datum_vars_by_datum,
                datum_handle,
                list_handle,
            ),
            initial_prototypes: self
                .initial_prototypes
                .iter()
                .map(|(path, datum)| (path.as_str().to_owned(), datum_handle(*datum)))
                .collect(),
            globals: self
                .globals
                .iter()
                .map(|(name, value)| (name.as_str().to_owned(), HeapSnapshotValue::from(value)))
                .collect(),
            compact_default_datums: sorted_datum_handles(&self.compact_default_datums),
            random_state: self.random_state,
            scheduler_tick: self.scheduler_tick,
            scheduler_sequence: self.scheduler_sequence,
            scheduled_spawns: self
                .scheduled_spawns
                .iter()
                .map(ScheduledSpawnSnapshot::from)
                .collect(),
            native_walks: {
                let mut walks = self
                    .native_walks
                    .iter()
                    .map(|(datum, walk)| (datum_handle(*datum), NativeWalkSnapshot::from(walk)))
                    .collect::<Vec<_>>();
                walks.sort_by_key(|(datum, _)| *datum);
                walks
            },
            last_animation_target: self
                .last_animation_target
                .as_ref()
                .map(HeapSnapshotValue::from),
            environment_overrides: self
                .environment_overrides
                .iter()
                .map(|(name, value)| (name.clone(), value.as_ref().map(HeapSnapshotValue::from)))
                .collect(),
            procedure_static_locals: self
                .procedure_static_locals
                .iter()
                .map(|(procedure, slots)| {
                    (
                        procedure.clone(),
                        slots
                            .iter()
                            .map(|(slot, value)| (*slot, HeapSnapshotValue::from(value)))
                            .collect(),
                    )
                })
                .collect(),
            external_timers: self
                .external_timers
                .iter()
                .map(|(name, started)| {
                    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    (name.clone(), elapsed)
                })
                .collect(),
            iconforge_jobs: self
                .iconforge_jobs
                .iter()
                .map(|(id, (complete, result))| (id.clone(), *complete, result.clone()))
                .collect(),
            iconforge_next_job: self.iconforge_next_job,
            iconforge_gags_configs: self
                .iconforge_gags_configs
                .iter()
                .map(|(name, path)| (name.clone(), path.to_string_lossy().into_owned()))
                .collect(),
            sql_jobs: self
                .sql_jobs
                .iter()
                .map(|(id, (complete, result))| (id.clone(), *complete, result.clone()))
                .collect(),
            sql_next_job: self.sql_next_job,
            default_world_area: self.default_world_area.map(datum_handle),
        }
    }

    /// Replaces mutable core state from a ready-world image while retaining the
    /// destination's immutable executable metadata.
    ///
    /// # Errors
    ///
    /// Returns a validation error for corrupt heap records, identifiers, or
    /// references to heap handles that are not live in the restored arena.
    pub fn restore_ready_world_core(
        &mut self,
        mut snapshot: ReadyWorldCoreSnapshot,
    ) -> Result<(), String> {
        self.assert_owner_thread();
        let heap_snapshot = std::mem::replace(&mut snapshot.heap, empty_heap_snapshot());
        let heap = ValueHeap::from_snapshot(heap_snapshot).map_err(|error| error.to_string())?;
        self.restore_ready_world_parts(snapshot, heap)
    }

    fn restore_ready_world_parts(
        &mut self,
        snapshot: ReadyWorldCoreSnapshot,
        heap: ValueHeap,
    ) -> Result<(), String> {
        validate_snapshot_handles(
            &heap,
            &snapshot.associative_lists,
            &snapshot.reference_lists,
            snapshot.global_vars_proxy,
            &snapshot.datum_vars_proxies,
            &snapshot.datum_vars_by_datum,
            &snapshot.initial_prototypes,
            &snapshot.compact_default_datums,
            snapshot.default_world_area,
        )?;

        self.heap = heap;
        self.associative_lists = snapshot
            .associative_lists
            .into_iter()
            .map(list_from_handle)
            .collect();
        self.reference_lists = snapshot
            .reference_lists
            .into_iter()
            .map(list_from_handle)
            .collect();
        self.savefiles = snapshot
            .savefiles
            .into_iter()
            .map(|(datum, cd, entries)| {
                let entries = entries
                    .into_iter()
                    .map(|(name, value)| Ok((name, value.into_value()?)))
                    .collect::<Result<HashMap<_, _>, ValueError>>()?;
                Ok((datum_from_handle(datum), SavefileState { entries, cd }))
            })
            .collect::<Result<_, ValueError>>()
            .map_err(|error| error.to_string())?;
        self.savefile_entries = snapshot
            .savefile_entries
            .into_iter()
            .map(|(entry, owner, name)| {
                (datum_from_handle(entry), (datum_from_handle(owner), name))
            })
            .collect();
        self.global_vars_proxy = snapshot.global_vars_proxy.map(list_from_handle);
        self.datum_vars_proxies = snapshot
            .datum_vars_proxies
            .into_iter()
            .map(|(list, datum)| (list_from_handle(list), datum_from_handle(datum)))
            .collect();
        self.datum_vars_by_datum = snapshot
            .datum_vars_by_datum
            .into_iter()
            .map(|(datum, list)| (datum_from_handle(datum), list_from_handle(list)))
            .collect();
        self.initial_prototypes = snapshot
            .initial_prototypes
            .into_iter()
            .map(|(path, datum)| Ok((TypePath::parse(&path)?, datum_from_handle(datum))))
            .collect::<Result<_, ValueError>>()
            .map_err(|error| error.to_string())?;
        self.initial_prototypes_initializing.clear();
        self.globals = snapshot
            .globals
            .into_iter()
            .map(|(name, value)| Ok((FieldName::parse(&name)?, value.into_value()?)))
            .collect::<Result<_, ValueError>>()
            .map_err(|error| error.to_string())?;
        self.compact_default_datums = snapshot
            .compact_default_datums
            .into_iter()
            .map(datum_from_handle)
            .collect();
        self.random_state = snapshot.random_state;
        self.scheduler_tick = snapshot.scheduler_tick;
        self.scheduler_sequence = snapshot.scheduler_sequence;
        self.scheduled_spawns = snapshot
            .scheduled_spawns
            .into_iter()
            .map(ScheduledSpawnSnapshot::into_runtime)
            .collect::<Result<_, ValueError>>()
            .map_err(|error| error.to_string())?;
        self.native_walks = snapshot
            .native_walks
            .into_iter()
            .map(|(datum, walk)| (datum_from_handle(datum), walk.into_runtime()))
            .collect();
        self.last_animation_target = snapshot
            .last_animation_target
            .map(HeapSnapshotValue::into_value)
            .transpose()
            .map_err(|error| error.to_string())?;
        self.environment_overrides = snapshot
            .environment_overrides
            .into_iter()
            .map(|(name, value)| Ok((name, value.map(HeapSnapshotValue::into_value).transpose()?)))
            .collect::<Result<_, ValueError>>()
            .map_err(|error| error.to_string())?;
        let restored_at = Instant::now();
        self.external_timers = snapshot
            .external_timers
            .into_iter()
            .map(|(name, elapsed)| {
                let started = restored_at
                    .checked_sub(Duration::from_nanos(elapsed))
                    .unwrap_or(restored_at);
                (name, started)
            })
            .collect();
        self.iconforge_jobs = snapshot
            .iconforge_jobs
            .into_iter()
            .map(|(id, complete, result)| (id, (complete, result)))
            .collect();
        self.iconforge_next_job = snapshot.iconforge_next_job;
        self.iconforge_gags_configs = snapshot
            .iconforge_gags_configs
            .into_iter()
            .map(|(name, path)| (name, PathBuf::from(path)))
            .collect();
        self.sql_jobs = snapshot
            .sql_jobs
            .into_iter()
            .map(|(id, complete, result)| (id, (complete, result)))
            .collect();
        self.sql_next_job = snapshot.sql_next_job;
        self.procedure_static_locals = snapshot
            .procedure_static_locals
            .into_iter()
            .map(|(procedure, slots)| {
                let slots = slots
                    .into_iter()
                    .map(|(slot, value)| Ok((slot, value.into_value()?)))
                    .collect::<Result<_, ValueError>>()?;
                Ok((procedure, slots))
            })
            .collect::<Result<_, ValueError>>()
            .map_err(|error| error.to_string())?;

        // Everything below is either derived from the restored heap or tied to
        // one host process. Rebuild/clear it instead of persisting addresses,
        // clocks, sockets, browser state, external jobs, or speculative caches.
        self.client_sessions.clear();
        self.interactive_local_clients.clear();
        self.local_client_outbound_events.clear();
        self.local_client_mobs.clear();
        self.local_client_commands.clear();
        self.pending_local_prompts.clear();
        self.dynamic_receiver_targets.clear();
        self.dynamic_callsite_targets.clear();
        self.declared_field_slots.clear();
        self.declared_field_quickening = DeclaredFieldQuickeningMetrics::default();
        self.clear_effective_initial_value_cache();
        self.scheduler_inflight.clear();
        self.scheduler_tick_started = None;
        self.atoms_profile = None;
        self.deleting_datums.clear();
        self.host_value_roots.clear();
        self.rebuild_world_geometry();
        self.default_world_area = snapshot.default_world_area.map(datum_from_handle);
        Ok(())
    }
}

fn ready_world_bincode() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(READY_WORLD_METADATA_LIMIT)
        .reject_trailing_bytes()
}

fn bincode_io_error(error: Box<bincode::ErrorKind>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn empty_heap_snapshot() -> ValueHeapSnapshot {
    ValueHeapSnapshot {
        datums: Vec::new(),
        datum_free: Vec::new(),
        lists: Vec::new(),
        list_free: Vec::new(),
    }
}

impl From<&ScheduledSpawn> for ScheduledSpawnSnapshot {
    fn from(spawn: &ScheduledSpawn) -> Self {
        Self {
            due_tick: spawn.due_tick,
            sequence: spawn.sequence,
            frames: spawn.frames.iter().map(CallFrameSnapshot::from).collect(),
        }
    }
}

impl ScheduledSpawnSnapshot {
    fn into_runtime(self) -> Result<ScheduledSpawn, ValueError> {
        Ok(ScheduledSpawn {
            due_tick: self.due_tick,
            sequence: self.sequence,
            frames: OwnedContinuation::new(
                VmContinuationId(self.sequence),
                self.frames
                    .into_iter()
                    .map(CallFrameSnapshot::into_runtime)
                    .collect::<Result<_, _>>()?,
            ),
        })
    }
}

impl From<&CallFrame> for CallFrameSnapshot {
    fn from(frame: &CallFrame) -> Self {
        let packed = frame
            .cold()
            .and_then(|cold| cold.packed_numeric_state.as_ref());
        Self {
            procedure: frame.procedure.0,
            instruction: frame.instruction,
            locals: packed.map_or_else(
                || snapshot_values(&frame.locals),
                |packed| {
                    snapshot_values(
                        &packed
                            .locals
                            .iter()
                            .copied()
                            .map(PackedValue::into_value)
                            .collect::<Vec<_>>(),
                    )
                },
            ),
            stack: packed.map_or_else(
                || snapshot_values(&frame.stack),
                |packed| {
                    snapshot_values(
                        &packed
                            .stack
                            .iter()
                            .copied()
                            .map(PackedValue::into_value)
                            .collect::<Vec<_>>(),
                    )
                },
            ),
            result: packed.map_or_else(
                || HeapSnapshotValue::from(&frame.result),
                |packed| HeapSnapshotValue::from(&packed.result.into_value()),
            ),
            src: HeapSnapshotValue::from(&frame.src),
            usr: HeapSnapshotValue::from(&frame.usr),
            arguments: snapshot_values(&frame.arguments),
            args_list: frame.args_list.map(list_handle),
            declared_argument_count: frame.declared_argument_count,
            supplied_parameters: frame.supplied_parameters.to_vec(),
            pending_argument_names: frame.pending_argument_names().cloned(),
            pending_argument_roots: snapshot_values(frame.pending_argument_roots()),
            retained_call_roots: snapshot_values(frame.retained_call_roots()),
            exception_handlers: frame
                .exception_handlers()
                .iter()
                .map(ExceptionHandlerSnapshot::from)
                .collect(),
            detached_waitfor: frame.detached_waitfor,
            caller_result_override: frame.caller_result_override().map(HeapSnapshotValue::from),
            engine_post_return: frame
                .engine_post_return()
                .map(CallFrameSnapshot::from)
                .map(Box::new),
            static_locals: frame.static_locals.to_vec(),
            shuttle_trace_target: frame.shuttle_trace_target().map(datum_handle),
            shuttle_trace_post_return: frame
                .shuttle_trace_post_return()
                .map(ShuttleTracePostReturnSnapshot::from),
            numeric_jit_state: frame.numeric_jit_state().map(NumericStateSnapshot::from),
        }
    }
}

impl CallFrameSnapshot {
    fn into_runtime(self) -> Result<CallFrame, ValueError> {
        let caller_result_override = self
            .caller_result_override
            .map(HeapSnapshotValue::into_value)
            .transpose()?;
        let engine_post_return = self
            .engine_post_return
            .map(|frame| frame.into_runtime().map(Box::new))
            .transpose()?;
        let numeric_jit_state = self
            .numeric_jit_state
            .map(NumericStateSnapshot::into_runtime);
        let cold = CallFrameCold {
            pending_argument_names: self.pending_argument_names,
            pending_argument_roots: restore_values(self.pending_argument_roots)?.into(),
            retained_call_roots: restore_values(self.retained_call_roots)?.into(),
            exception_handlers: self
                .exception_handlers
                .into_iter()
                .map(ExceptionHandlerSnapshot::into_runtime)
                .collect(),
            shuttle_trace_target: self.shuttle_trace_target.map(datum_from_handle),
            shuttle_trace_post_return: self
                .shuttle_trace_post_return
                .map(ShuttleTracePostReturnSnapshot::into_runtime),
            caller_result_override,
            engine_post_return,
            numeric_jit_state,
            ..CallFrameCold::default()
        };
        let cold = (!cold.is_empty()).then(|| Box::new(cold));
        Ok(CallFrame {
            procedure: ProcedureId(self.procedure),
            instruction: self.instruction,
            locals: restore_values(self.locals)?.into(),
            stack: restore_values(self.stack)?.into(),
            result: self.result.into_value()?,
            src: self.src.into_value()?,
            usr: self.usr.into_value()?,
            arguments: restore_values(self.arguments)?.into(),
            args_list: self.args_list.map(list_from_handle),
            declared_argument_count: self.declared_argument_count,
            supplied_parameters: self.supplied_parameters.into(),
            cold,
            detached_waitfor: self.detached_waitfor,
            static_locals: self.static_locals.into(),
            atoms_profile_entry_counted: false,
            atoms_profile_root: false,
            tgm_profile_root: false,
        })
    }
}

impl From<&ExceptionHandler> for ExceptionHandlerSnapshot {
    fn from(handler: &ExceptionHandler) -> Self {
        Self {
            start: handler.start,
            end: handler.end,
            catch: handler.catch,
            local: handler.local,
            stack_depth: handler.stack_depth,
        }
    }
}

impl ExceptionHandlerSnapshot {
    fn into_runtime(self) -> ExceptionHandler {
        ExceptionHandler {
            start: self.start,
            end: self.end,
            catch: self.catch,
            local: self.local,
            stack_depth: self.stack_depth,
        }
    }
}

impl From<&ShuttleTracePostReturn> for ShuttleTracePostReturnSnapshot {
    fn from(value: &ShuttleTracePostReturn) -> Self {
        match value {
            ShuttleTracePostReturn::AtmosInit => Self::AtmosInit,
            ShuttleTracePostReturn::NullifyNode { slot } => Self::NullifyNode { slot: *slot },
        }
    }
}

impl ShuttleTracePostReturnSnapshot {
    fn into_runtime(self) -> ShuttleTracePostReturn {
        match self {
            Self::AtmosInit => ShuttleTracePostReturn::AtmosInit,
            Self::NullifyNode { slot } => ShuttleTracePostReturn::NullifyNode { slot },
        }
    }
}

impl From<&NumericExecutionState> for NumericStateSnapshot {
    fn from(state: &NumericExecutionState) -> Self {
        Self {
            locals: state.locals.to_vec(),
            stack: state.stack.to_vec(),
            fields: state.fields.to_vec(),
            dirty_fields: state.dirty_fields,
            action_bits: state.action_bits,
            instruction: state.instruction,
        }
    }
}

impl NumericStateSnapshot {
    fn into_runtime(self) -> NumericExecutionState {
        NumericExecutionState {
            locals: self.locals.into(),
            stack: self.stack.into(),
            fields: self.fields.into(),
            dirty_fields: self.dirty_fields,
            action_bits: self.action_bits,
            instruction: self.instruction,
        }
    }
}

impl From<&NativeWalk> for NativeWalkSnapshot {
    fn from(walk: &NativeWalk) -> Self {
        Self {
            due_tick: walk.due_tick,
            sequence: walk.sequence,
            lag: walk.lag,
            kind: match walk.kind {
                NativeWalkKind::Direction(direction) => {
                    NativeWalkKindSnapshot::Direction(direction)
                }
                NativeWalkKind::Random => NativeWalkKindSnapshot::Random,
                NativeWalkKind::Towards(target) => {
                    NativeWalkKindSnapshot::Towards(datum_handle(target))
                }
                NativeWalkKind::To { target, minimum } => NativeWalkKindSnapshot::To {
                    target: datum_handle(target),
                    minimum,
                },
                NativeWalkKind::Away { target, maximum } => NativeWalkKindSnapshot::Away {
                    target: datum_handle(target),
                    maximum,
                },
            },
        }
    }
}

impl NativeWalkSnapshot {
    fn into_runtime(self) -> NativeWalk {
        NativeWalk {
            due_tick: self.due_tick,
            sequence: self.sequence,
            lag: self.lag,
            kind: match self.kind {
                NativeWalkKindSnapshot::Direction(direction) => {
                    NativeWalkKind::Direction(direction)
                }
                NativeWalkKindSnapshot::Random => NativeWalkKind::Random,
                NativeWalkKindSnapshot::Towards(target) => {
                    NativeWalkKind::Towards(datum_from_handle(target))
                }
                NativeWalkKindSnapshot::To { target, minimum } => NativeWalkKind::To {
                    target: datum_from_handle(target),
                    minimum,
                },
                NativeWalkKindSnapshot::Away { target, maximum } => NativeWalkKind::Away {
                    target: datum_from_handle(target),
                    maximum,
                },
            },
        }
    }
}

fn snapshot_values(values: &[Value]) -> Vec<HeapSnapshotValue> {
    values.iter().map(HeapSnapshotValue::from).collect()
}

fn restore_values(values: Vec<HeapSnapshotValue>) -> Result<Vec<Value>, ValueError> {
    values
        .into_iter()
        .map(HeapSnapshotValue::into_value)
        .collect()
}

fn validate_frame(module: &Module, frame: &CallFrameSnapshot, depth: usize) -> Result<(), String> {
    if depth > 1_024 {
        return Err("ready-world engine-return frame nesting exceeds 1024".into());
    }
    let procedure = ProcedureId(frame.procedure);
    let program = module.procedure(procedure).ok_or_else(|| {
        format!(
            "ready-world frame references missing procedure {}",
            frame.procedure
        )
    })?;
    if frame.instruction >= program.instructions.len() {
        return Err(format!(
            "ready-world frame instruction {} exceeds procedure {} length {}",
            frame.instruction,
            frame.procedure,
            program.instructions.len()
        ));
    }
    if frame.locals.len() != program.local_count {
        return Err(format!(
            "ready-world frame for procedure {} has {} locals; executable expects {}",
            frame.procedure,
            frame.locals.len(),
            program.local_count
        ));
    }
    for handler in &frame.exception_handlers {
        if handler.start > handler.end
            || handler.end > program.instructions.len()
            || handler.catch >= program.instructions.len()
            || handler.stack_depth > frame.stack.len()
            || handler
                .local
                .is_some_and(|local| local as usize >= frame.locals.len())
        {
            return Err(format!(
                "ready-world frame for procedure {} has an invalid exception handler",
                frame.procedure
            ));
        }
    }
    if let Some(next) = frame.engine_post_return.as_deref() {
        validate_frame(module, next, depth + 1)?;
    }
    Ok(())
}

fn validate_snapshot_handles(
    heap: &ValueHeap,
    associative_lists: &[ListHandle],
    reference_lists: &[ListHandle],
    global_vars_proxy: Option<ListHandle>,
    datum_vars_proxies: &[(ListHandle, DatumHandle)],
    datum_vars_by_datum: &[(DatumHandle, ListHandle)],
    initial_prototypes: &[(String, DatumHandle)],
    compact_default_datums: &[DatumHandle],
    default_world_area: Option<DatumHandle>,
) -> Result<(), String> {
    let validate_datum = |handle: DatumHandle| {
        heap.datum(datum_from_handle(handle))
            .map(|_| ())
            .map_err(|error| error.to_string())
    };
    let validate_list = |handle: ListHandle| {
        heap.list(list_from_handle(handle))
            .map(|_| ())
            .map_err(|error| error.to_string())
    };
    for handle in associative_lists.iter().chain(reference_lists).copied() {
        validate_list(handle)?;
    }
    for handle in initial_prototypes
        .iter()
        .map(|(_, handle)| *handle)
        .chain(compact_default_datums.iter().copied())
        .chain(default_world_area)
    {
        validate_datum(handle)?;
    }
    if let Some(handle) = global_vars_proxy {
        validate_list(handle)?;
    }
    for (list, datum) in datum_vars_proxies {
        validate_list(*list)?;
        validate_datum(*datum)?;
    }
    for (datum, list) in datum_vars_by_datum {
        validate_datum(*datum)?;
        validate_list(*list)?;
    }
    Ok(())
}

fn datum_handle(id: DatumId) -> DatumHandle {
    (id.index(), id.generation())
}

fn list_handle(id: ListId) -> ListHandle {
    (id.index(), id.generation())
}

fn datum_from_handle((index, generation): DatumHandle) -> DatumId {
    DatumId::from_parts(index, generation)
}

fn list_from_handle((index, generation): ListHandle) -> ListId {
    ListId::from_parts(index, generation)
}

fn sorted_datum_handles(values: &HashSet<DatumId>) -> Vec<DatumHandle> {
    let mut values = values.iter().copied().map(datum_handle).collect::<Vec<_>>();
    values.sort_unstable();
    values
}

fn sorted_list_handles(values: &HashSet<ListId>) -> Vec<ListHandle> {
    let mut values = values.iter().copied().map(list_handle).collect::<Vec<_>>();
    values.sort_unstable();
    values
}

fn sorted_handle_map<K, V, SK: Ord, SV>(
    values: &HashMap<K, V>,
    key: impl Fn(K) -> SK,
    value: impl Fn(V) -> SV,
) -> Vec<(SK, SV)>
where
    K: Copy,
    V: Copy,
{
    let mut values = values
        .iter()
        .map(|(left, right)| (key(*left), value(*right)))
        .collect::<Vec<_>>();
    values.sort_by(|left, right| left.0.cmp(&right.0));
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_snapshot_restores_mutable_state_and_rebuilds_geometry() {
        let mut source = ExecutionState::new();
        let turf = source
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/open").unwrap());
        for (name, value) in [("x", 2.0), ("y", 3.0), ("z", 1.0)] {
            source
                .heap_mut()
                .set_datum_field(turf, FieldName::parse(name).unwrap(), Value::number(value))
                .unwrap();
        }
        let list = source.heap_mut().allocate_list();
        source
            .heap_mut()
            .list_mut(list)
            .unwrap()
            .add(Value::Datum(turf));
        source.associative_lists.insert(list);
        source.set_global(FieldName::parse("root").unwrap(), Value::List(list));
        source.random_state = 123;
        source.scheduler_tick = 456;
        source.scheduler_sequence = 789;
        let program = Program {
            wait_for: true,
            parameter_count: 1,
            parameter_names: vec!["item".into()],
            verb_parameter_types: Vec::new(),
            verb_name: None,
            local_count: 2,
            source_spans: vec![SourceSpan::new(0, 1); 2],
            instructions: vec![Instruction::PushNull, Instruction::Return],
        };
        let mut frame = make_frame(
            ProcedureId(7),
            &program,
            &[Value::List(list)],
            &ExecutionContext::new(Value::Datum(turf), Value::Null),
        );
        frame.instruction = 1;
        frame.stack.push(Value::number(9.0));
        source.scheduled_spawns.push(ScheduledSpawn {
            due_tick: 500,
            sequence: 42,
            frames: OwnedContinuation::new(VmContinuationId(42), vec![frame]),
        });
        source.rebuild_world_geometry();

        let snapshot = source.capture_ready_world_core();
        let module = Module {
            identity: crate::bytecode::ModuleIdentity(1),
            procedures: (0..8)
                .map(|index| {
                    Arc::new(if index == 7 {
                        program.clone()
                    } else {
                        Program {
                            wait_for: true,
                            parameter_count: 0,
                            parameter_names: Vec::new(),
                            verb_parameter_types: Vec::new(),
                            verb_name: None,
                            local_count: 0,
                            source_spans: vec![SourceSpan::new(0, 1)],
                            instructions: vec![Instruction::Return],
                        }
                    })
                })
                .collect(),
            paths: (0..8).map(|index| format!("/proc/test_{index}")).collect(),
            names: HashMap::new(),
            dynamic_names: HashMap::new(),
            deferred: Arc::new(HashMap::new()),
            procedure_types: Vec::new(),
            initializer_call_names: None,
            compact_wordcode: Default::default(),
            semantic_digests: Default::default(),
        };
        snapshot.validate_module(&module).unwrap();
        let mut restored = ExecutionState::new();
        restored.restore_ready_world_core(snapshot).unwrap();
        assert_eq!(
            restored.global(&FieldName::parse("root").unwrap()),
            Some(&Value::List(list))
        );
        assert!(restored.associative_lists.contains(&list));
        assert_eq!(restored.random_state, 123);
        assert_eq!(restored.scheduler_tick(), 456);
        assert_eq!(restored.world_turfs.get(&(2, 3, 1)), Some(&turf));
        assert_eq!(restored.scheduled_spawns.len(), 1);
        assert_eq!(restored.scheduled_spawns[0].frames.id.get(), 42);
        let restored_frame = &restored.scheduled_spawns[0].frames[0];
        assert_eq!(restored_frame.procedure, ProcedureId(7));
        assert_eq!(restored_frame.instruction, 1);
        assert_eq!(restored_frame.locals[0], Value::List(list));
        assert_eq!(restored_frame.stack[0], Value::number(9.0));

        let mut encoded = Vec::new();
        source.write_ready_world_snapshot_to(&mut encoded).unwrap();
        let mut streamed = ExecutionState::new();
        streamed
            .restore_ready_world_snapshot_from(&mut encoded.as_slice(), &module)
            .unwrap();
        assert_eq!(
            streamed.global(&FieldName::parse("root").unwrap()),
            Some(&Value::List(list))
        );
        assert_eq!(streamed.scheduled_spawns.len(), 1);
        assert_eq!(streamed.scheduled_spawns[0].frames.id.get(), 42);
        assert_eq!(streamed.scheduled_spawns[0].frames[0].instruction, 1);

        let restored_on_worker = std::thread::spawn(move || {
            let mut restored = ExecutionState::new();
            restored
                .restore_ready_world_snapshot_from(&mut encoded.as_slice(), &module)
                .unwrap();
            restored.set_global(
                FieldName::parse("worker_owned").unwrap(),
                Value::number(1.0),
            );
            restored
                .global(&FieldName::parse("worker_owned").unwrap())
                .cloned()
        });
        assert_eq!(
            restored_on_worker
                .join()
                .expect("worker-owned snapshot restore should succeed"),
            Some(Value::number(1.0))
        );
    }
}
