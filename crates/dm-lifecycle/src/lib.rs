//! Deterministic, non-executing lifecycle resolution and initialization plans.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_core::{FileId, SourceSpan};
use dm_object_tree::{NodeId, NodeKind};
use dm_runtime::RuntimeImage;
use dm_semantics::{Procedure, ProcedureId, ProcedureImplementationId, ProcedureRegistry};
use dm_value::{DatumId, TypePath, Value};
use dm_vm::{ExecutionContext, ExecutionState, RuntimeError, execute_module_in_context};
use dm_world::{AtomCategory, InitializerResolution, WorldAllocation, WorldCoordinate, WorldPlan};

/// Lifecycle entry points resolved for every runtime type.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LifecycleKind {
    /// Construction hook.
    New,
    /// Primary atom initialization hook.
    Initialize,
    /// Deferred atom initialization hook.
    LateInitialize,
    /// Destruction and cleanup hook.
    Destroy,
}

impl LifecycleKind {
    const ALL: [Self; 4] = [
        Self::New,
        Self::Initialize,
        Self::LateInitialize,
        Self::Destroy,
    ];

    const INITIALIZATION: [Self; 3] = [Self::New, Self::Initialize, Self::LateInitialize];

    const fn procedure_name(self) -> &'static str {
        match self {
            Self::New => "New",
            Self::Initialize => "Initialize",
            Self::LateInitialize => "LateInitialize",
            Self::Destroy => "Destroy",
        }
    }
}

/// Original source location of a resolved lifecycle implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleSource {
    /// Physical project file identity.
    pub file_id: FileId,
    /// Project-relative source path.
    pub path: String,
    /// Original source range of the implementation header.
    pub span: SourceSpan,
    /// Expanded declaration ordinal.
    pub ordinal: usize,
}

/// Effective implementation selected for one type and lifecycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleTarget {
    /// Canonical procedure identity.
    pub procedure: ProcedureId,
    /// Exact effective body identity.
    pub implementation: ProcedureImplementationId,
    /// Canonical procedure path.
    pub procedure_path: String,
    /// Type that owns the selected procedure node.
    pub declaring_type: String,
    /// Whether dispatch inherited this target from an ancestor type.
    pub inherited: bool,
    /// Source definition selected for dispatch.
    pub source: LifecycleSource,
}

/// Structural reason a lifecycle procedure cannot be targeted.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LifecycleTargetIssueKind {
    /// A procedure node has no effective implementation.
    MissingEffectiveTarget,
    /// The effective identity is absent from the registry.
    MissingImplementation,
    /// The implementation's syntax definition is unavailable.
    MissingSourceDefinition,
    /// A runtime type has no corresponding compiler type node.
    MissingCompilerType,
}

/// Unsupported lifecycle resolution retained on the affected type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleTargetIssue {
    /// Stable issue category.
    pub kind: LifecycleTargetIssueKind,
    /// Human-readable detail.
    pub message: String,
    /// Canonical procedure path, when resolution reached one.
    pub procedure_path: Option<String>,
}

/// Effective state of one lifecycle entry point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LifecycleResolution {
    /// No implementation exists in the effective inheritance chain.
    Absent,
    /// An exact effective implementation and source were resolved.
    Resolved(LifecycleTarget),
    /// A procedure was present but its target metadata was incomplete.
    Unsupported(LifecycleTargetIssue),
}

/// All four effective lifecycle entry points for one type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleTargets {
    /// Effective `New()` target.
    pub new_target: LifecycleResolution,
    /// Effective `Initialize()` target.
    pub initialize: LifecycleResolution,
    /// Effective `LateInitialize()` target.
    pub late_initialize: LifecycleResolution,
    /// Effective `Destroy()` target.
    pub destroy: LifecycleResolution,
}

impl LifecycleTargets {
    /// Returns the requested lifecycle resolution.
    #[must_use]
    pub const fn get(&self, kind: LifecycleKind) -> &LifecycleResolution {
        match kind {
            LifecycleKind::New => &self.new_target,
            LifecycleKind::Initialize => &self.initialize,
            LifecycleKind::LateInitialize => &self.late_initialize,
            LifecycleKind::Destroy => &self.destroy,
        }
    }
}

/// Runtime type metadata paired with its effective lifecycle dispatch targets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeLifecycle {
    /// Compiler object-tree identity.
    pub node: NodeId,
    /// Canonical runtime type path.
    pub path: String,
    /// Effective runtime parent path.
    pub parent: Option<String>,
    /// Effective lifecycle targets.
    pub targets: LifecycleTargets,
}

/// Stable diagnostic category produced while indexing or planning.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LifecycleDiagnosticKind {
    /// Runtime and compiler type metadata disagree.
    MissingCompilerType,
    /// A present lifecycle procedure has incomplete target metadata.
    UnsupportedLifecycleTarget,
    /// Runtime materialization retained a non-constant initializer.
    UnsupportedInitializer,
    /// The canonical `/world` type is unavailable.
    MissingWorldType,
    /// A resolved map type is absent from the lifecycle index.
    MissingTypeLifecycle,
    /// A planned cell references no retained key template.
    MissingMapTemplate,
    /// A map initializer path is unknown or is not a type.
    UnsupportedMapInitializer,
}

/// Source- and coordinate-aware recoverable lifecycle diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleDiagnostic {
    /// Stable diagnostic category.
    pub kind: LifecycleDiagnosticKind,
    /// Human-readable detail.
    pub message: String,
    /// Canonical affected type or variable path.
    pub path: Option<String>,
    /// Source file path, when available.
    pub source_path: Option<String>,
    /// Relevant source span, when available.
    pub span: Option<SourceSpan>,
    /// Map coordinate, when the issue affects one placement.
    pub coordinate: Option<WorldCoordinate>,
}

/// Effective lifecycle metadata for every canonical runtime type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleIndex {
    types: Vec<TypeLifecycle>,
    by_node: BTreeMap<NodeId, usize>,
    by_path: BTreeMap<String, usize>,
    diagnostics: Vec<LifecycleDiagnostic>,
}

impl LifecycleIndex {
    /// Builds lifecycle dispatch metadata without compiling or executing bodies.
    #[must_use]
    pub fn build(
        compilation: &Compilation,
        procedures: &ProcedureRegistry,
        runtime: &RuntimeImage,
    ) -> Self {
        let tree = compilation.code_tree();
        let compiler_types: BTreeMap<_, _> = tree
            .nodes()
            .iter()
            .filter(|node| node.kind == NodeKind::Type)
            .map(|node| (node.path.to_string(), node.id))
            .collect();
        let direct = direct_lifecycle_procedures(procedures);
        let mut diagnostics = Vec::new();
        let mut types = Vec::new();
        for (runtime_path, runtime_type) in runtime.types() {
            let path = runtime_path.to_string();
            let Some(node) = compiler_types.get(&path).copied() else {
                diagnostics.push(LifecycleDiagnostic {
                    kind: LifecycleDiagnosticKind::MissingCompilerType,
                    message: format!("runtime type {path} is absent from the compiler tree"),
                    path: Some(path),
                    source_path: None,
                    span: None,
                    coordinate: None,
                });
                continue;
            };
            let targets = LifecycleTargets {
                new_target: resolve_target(
                    compilation,
                    procedures,
                    &direct,
                    node,
                    LifecycleKind::New,
                ),
                initialize: resolve_target(
                    compilation,
                    procedures,
                    &direct,
                    node,
                    LifecycleKind::Initialize,
                ),
                late_initialize: resolve_target(
                    compilation,
                    procedures,
                    &direct,
                    node,
                    LifecycleKind::LateInitialize,
                ),
                destroy: resolve_target(
                    compilation,
                    procedures,
                    &direct,
                    node,
                    LifecycleKind::Destroy,
                ),
            };
            collect_target_diagnostics(&path, &targets, &mut diagnostics);
            types.push(TypeLifecycle {
                node,
                path,
                parent: runtime_type.parent().map(ToString::to_string),
                targets,
            });
        }
        let by_node = types
            .iter()
            .enumerate()
            .map(|(index, lifecycle)| (lifecycle.node, index))
            .collect();
        let by_path = types
            .iter()
            .enumerate()
            .map(|(index, lifecycle)| (lifecycle.path.clone(), index))
            .collect();
        Self {
            types,
            by_node,
            by_path,
            diagnostics,
        }
    }

    /// Returns types in canonical runtime path order.
    #[must_use]
    pub fn types(&self) -> &[TypeLifecycle] {
        &self.types
    }

    /// Looks up lifecycle metadata by compiler type identity.
    #[must_use]
    pub fn find_node(&self, node: NodeId) -> Option<&TypeLifecycle> {
        self.types.get(*self.by_node.get(&node)?)
    }

    /// Looks up lifecycle metadata by canonical type path.
    #[must_use]
    pub fn find_path(&self, path: &str) -> Option<&TypeLifecycle> {
        self.types.get(*self.by_path.get(path)?)
    }

    /// Returns recoverable index diagnostics in deterministic encounter order.
    #[must_use]
    pub fn diagnostics(&self) -> &[LifecycleDiagnostic] {
        &self.diagnostics
    }

    fn index_for_node(&self, node: NodeId) -> Option<usize> {
        self.by_node.get(&node).copied()
    }
}

/// Runtime-image work represented by the first initialization event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalInitialization {
    /// Explicit initializer steps processed by the runtime image.
    pub initializer_steps: usize,
    /// Constant values already materialized without executing DM.
    pub constants_materialized: usize,
    /// Materialized global and type-static slots.
    pub runtime_variables: usize,
    /// Initializers deferred to a future execution phase.
    pub unsupported_initializers: usize,
}

/// Map-source context retained for one atom placement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MapPlacementContext {
    /// Map path supplied by the caller.
    pub map_path: String,
    /// Cell key that supplied the initializer template.
    pub key: String,
    /// Expanded stable world coordinate.
    pub coordinate: WorldCoordinate,
    /// Source range of the initializer in the map key definition.
    pub initializer_span: SourceSpan,
    /// Source range of the coordinate block placing the template.
    pub block_span: SourceSpan,
}

/// One resolved map atom placement and its lifecycle type metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedAtom {
    /// Index into [`LifecycleIndex::types`].
    pub type_index: usize,
    /// Canonical type path.
    pub type_path: String,
    /// World atom category inferred by the map planner.
    pub category: AtomCategory,
    /// Exact map placement source and coordinate context.
    pub placement: MapPlacementContext,
}

/// Subject receiving a planned initialization event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventSubject {
    /// Runtime global/default image materialization.
    Globals,
    /// The singleton `/world` object.
    World,
    /// One entry in [`InitializationPlan::map_atoms`].
    MapAtom(usize),
}

/// One non-executing event in deterministic initialization order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitializationEvent {
    /// Materialize constants and retain deferred global/default work.
    Globals,
    /// Invoke a resolved lifecycle body in a future execution phase.
    Lifecycle {
        /// Object receiving the call.
        subject: EventSubject,
        /// Lifecycle hook to invoke.
        kind: LifecycleKind,
        /// Index into [`LifecycleIndex::types`].
        type_index: usize,
    },
}

/// Globals, world, and map-placement initialization plan with no executed code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializationPlan {
    /// Runtime global/default materialization summary.
    pub globals: GlobalInitialization,
    /// Lifecycle type index for `/world`, when available.
    pub world_type: Option<usize>,
    /// Resolved map placements in cell and initializer order.
    pub map_atoms: Vec<PlannedAtom>,
    /// Globals-first, world-New, map-New/Initialize/LateInitialize events.
    pub events: Vec<InitializationEvent>,
    /// Recoverable source-aware diagnostics.
    pub diagnostics: Vec<LifecycleDiagnostic>,
}

impl InitializationPlan {
    /// Counts map atom placements with each resolved lifecycle target.
    #[must_use]
    pub fn map_lifecycle_counts(&self, index: &LifecycleIndex) -> BTreeMap<LifecycleKind, usize> {
        let mut counts = BTreeMap::new();
        for atom in &self.map_atoms {
            let Some(lifecycle) = index.types.get(atom.type_index) else {
                continue;
            };
            for kind in LifecycleKind::ALL {
                if matches!(
                    lifecycle.targets.get(kind),
                    LifecycleResolution::Resolved(_)
                ) {
                    *counts.entry(kind).or_default() += 1;
                }
            }
        }
        counts
    }
}

/// Builds a deterministic globals/world/map lifecycle plan without execution.
#[must_use]
pub fn build_initialization_plan(
    runtime: &RuntimeImage,
    index: &LifecycleIndex,
    world: &WorldPlan,
    map_path: impl Into<String>,
) -> InitializationPlan {
    let map_path = map_path.into();
    let globals = global_initialization(runtime);
    let mut diagnostics = initialization_diagnostics(runtime, index);
    let world_type = index.by_path.get("/world").copied();
    if world_type.is_none() {
        diagnostics.push(LifecycleDiagnostic {
            kind: LifecycleDiagnosticKind::MissingWorldType,
            message: "runtime lifecycle index contains no /world type".to_owned(),
            path: Some("/world".to_owned()),
            source_path: None,
            span: None,
            coordinate: None,
        });
    }
    let map_atoms = plan_map_atoms(index, world, &map_path, &mut diagnostics);
    let events = initialization_events(index, world_type, &map_atoms);
    InitializationPlan {
        globals,
        world_type,
        map_atoms,
        events,
        diagnostics,
    }
}

/// One successfully invoked lifecycle hook.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutedLifecycleEvent {
    /// Planned event that selected the hook.
    pub event: InitializationEvent,
    /// Live datum passed as `src` to the hook.
    pub datum: DatumId,
    /// Canonical selected procedure path.
    pub procedure_path: String,
    /// Result value returned by the hook.
    pub result: Value,
}

/// Deterministic result of executing a planned initialization sequence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InitializationExecution {
    /// The allocated singleton `/world` datum.
    pub world: Option<DatumId>,
    /// Hooks executed in event order.
    pub events: Vec<ExecutedLifecycleEvent>,
    /// Repeated map placements sharing an already initialized datum.
    pub duplicate_map_events: usize,
}

/// Failure while binding or executing a planned lifecycle hook.
#[derive(Debug)]
pub enum InitializationExecutionError {
    /// Selected lifecycle bodies could not be lowered to the reference VM.
    Compile(dm_vm::CompileError),
    /// A planned map atom has no matching allocated datum.
    MissingMapDatum {
        /// Index into [`InitializationPlan::map_atoms`].
        atom_index: usize,
        /// Canonical type expected by the lifecycle plan.
        path: String,
    },
    /// Lifecycle metadata did not retain an executable target.
    MissingTarget {
        /// Index into [`LifecycleIndex::types`].
        type_index: usize,
        /// Requested lifecycle phase.
        kind: LifecycleKind,
    },
    /// A selected semantic body was absent from the compiled VM module.
    MissingVmTarget {
        /// Canonical selected procedure path.
        procedure_path: String,
    },
    /// The fixed `/world` path could not be represented by runtime values.
    WorldPath(dm_value::ValueError),
    /// A world lifecycle event was present but no singleton datum was allocated.
    MissingWorldDatum,
    /// The runtime image cannot allocate the singleton `/world` datum.
    WorldAllocation(dm_runtime::RuntimeImageError),
    /// VM execution failed with its original source-mapped call stack.
    Runtime {
        /// Event being executed.
        event: InitializationEvent,
        /// Source-selected lifecycle target.
        target: Box<LifecycleTarget>,
        /// Original VM failure.
        error: Box<RuntimeError>,
    },
}

impl std::fmt::Display for InitializationExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(error) => write!(formatter, "lifecycle compilation failed: {error}"),
            Self::MissingMapDatum { atom_index, path } => write!(
                formatter,
                "map lifecycle atom {atom_index} ({path}) has no allocated datum"
            ),
            Self::MissingTarget { type_index, kind } => {
                write!(
                    formatter,
                    "lifecycle target {kind:?} is missing for type {type_index}"
                )
            }
            Self::MissingVmTarget { procedure_path } => {
                write!(
                    formatter,
                    "lifecycle VM target is missing for {procedure_path}"
                )
            }
            Self::WorldPath(error) => write!(formatter, "world type path is invalid: {error}"),
            Self::MissingWorldDatum => {
                formatter.write_str("world lifecycle event has no allocated datum")
            }
            Self::WorldAllocation(error) => write!(formatter, "world allocation failed: {error}"),
            Self::Runtime { target, error, .. } => {
                write!(
                    formatter,
                    "lifecycle {} failed: {error}",
                    target.procedure_path
                )
            }
        }
    }
}

impl std::error::Error for InitializationExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compile(error) => Some(error),
            Self::WorldPath(error) => Some(error),
            Self::WorldAllocation(error) => Some(error),
            Self::Runtime { error, .. } => Some(error),
            Self::MissingMapDatum { .. }
            | Self::MissingTarget { .. }
            | Self::MissingVmTarget { .. }
            | Self::MissingWorldDatum => None,
        }
    }
}

/// Executes `New`, `Initialize`, and `LateInitialize` for allocated map atoms.
///
/// The caller first builds an [`InitializationPlan`] and materializes the same
/// [`WorldPlan`] with [`dm_world::allocate_world`]. Hooks use each live datum as
/// `src`, execute in plan order, and share one mutable VM heap. Repeated map
/// placements referring to one shared area datum run each hook once.
///
/// # Errors
///
/// Returns a source-aware error when a planned target cannot be compiled,
/// bound to an allocation, or executed.
pub fn execute_initialization_plan(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
) -> Result<InitializationExecution, InitializationExecutionError> {
    let bindings = map_datum_bindings(plan, allocation, runtime);
    let targets = plan
        .events
        .iter()
        .filter_map(|event| event_target(*event, index).map(|target| target.implementation))
        .collect::<BTreeSet<_>>();
    let executable = procedures
        .compile_vm_implementations(compilation, targets)
        .map_err(InitializationExecutionError::Compile)?;
    let world = if plan.events.iter().any(|event| {
        matches!(
            event,
            InitializationEvent::Lifecycle {
                subject: EventSubject::World,
                ..
            }
        )
    }) {
        Some(
            runtime
                .allocate_datum(
                    &TypePath::parse("/world").map_err(InitializationExecutionError::WorldPath)?,
                )
                .map_err(InitializationExecutionError::WorldAllocation)?,
        )
    } else {
        None
    };

    let mut state = ExecutionState::from_heap(std::mem::take(runtime.heap_mut()));
    let execution = (|| {
        let mut result = InitializationExecution {
            world,
            ..InitializationExecution::default()
        };
        let mut seen = BTreeSet::new();
        for event in &plan.events {
            let InitializationEvent::Lifecycle { subject, .. } = *event else {
                continue;
            };
            let datum = match subject {
                EventSubject::World => {
                    world.ok_or(InitializationExecutionError::MissingWorldDatum)?
                }
                EventSubject::MapAtom(atom_index) => bindings
                    .get(atom_index)
                    .and_then(|datum| *datum)
                    .ok_or_else(|| InitializationExecutionError::MissingMapDatum {
                        atom_index,
                        path: plan.map_atoms[atom_index].type_path.clone(),
                    })?,
                EventSubject::Globals => continue,
            };
            if matches!(subject, EventSubject::MapAtom(_))
                && !seen.insert((datum, event_kind(*event)))
            {
                result.duplicate_map_events += 1;
                continue;
            }
            let target = event_target(*event, index).ok_or_else(|| {
                InitializationExecutionError::MissingTarget {
                    type_index: event_type_index(*event),
                    kind: event_kind(*event),
                }
            })?;
            let entry = executable
                .implementation(target.implementation)
                .ok_or_else(|| InitializationExecutionError::MissingVmTarget {
                    procedure_path: target.procedure_path.clone(),
                })?;
            let value = execute_module_in_context(
                executable.module(),
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            )
            .map_err(|error| InitializationExecutionError::Runtime {
                event: *event,
                target: Box::new(target.clone()),
                error: Box::new(error),
            })?;
            result.events.push(ExecutedLifecycleEvent {
                event: *event,
                datum,
                procedure_path: target.procedure_path.clone(),
                result: value,
            });
        }
        Ok(result)
    })();
    *runtime.heap_mut() = state.into_heap();
    execution
}

fn event_kind(event: InitializationEvent) -> LifecycleKind {
    match event {
        InitializationEvent::Lifecycle { kind, .. } => kind,
        InitializationEvent::Globals => unreachable!("global initialization has no lifecycle kind"),
    }
}

fn event_type_index(event: InitializationEvent) -> usize {
    match event {
        InitializationEvent::Lifecycle { type_index, .. } => type_index,
        InitializationEvent::Globals => unreachable!("global initialization has no lifecycle type"),
    }
}

fn event_target(event: InitializationEvent, index: &LifecycleIndex) -> Option<&LifecycleTarget> {
    let InitializationEvent::Lifecycle {
        kind, type_index, ..
    } = event
    else {
        return None;
    };
    match index.types.get(type_index)?.targets.get(kind) {
        LifecycleResolution::Resolved(target) => Some(target),
        LifecycleResolution::Absent | LifecycleResolution::Unsupported(_) => None,
    }
}

fn map_datum_bindings(
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &RuntimeImage,
) -> Vec<Option<DatumId>> {
    let by_coordinate: BTreeMap<_, Vec<DatumId>> = allocation
        .snapshots()
        .iter()
        .map(|snapshot| (snapshot.coordinate, snapshot.source_order.clone()))
        .collect();
    let mut positions = BTreeMap::<WorldCoordinate, usize>::new();
    plan.map_atoms
        .iter()
        .map(|atom| {
            if atom.category == AtomCategory::OtherType {
                return None;
            }
            let position = positions.entry(atom.placement.coordinate).or_default();
            let datum = by_coordinate
                .get(&atom.placement.coordinate)
                .and_then(|datums| datums.get(*position))
                .copied();
            let matches_type = datum.is_some_and(|datum| {
                runtime
                    .heap()
                    .datum(datum)
                    .is_ok_and(|record| record.type_path().as_str() == atom.type_path)
            });
            if matches_type {
                *position += 1;
                datum
            } else {
                None
            }
        })
        .collect()
}

fn global_initialization(runtime: &RuntimeImage) -> GlobalInitialization {
    let stats = runtime.stats();
    GlobalInitialization {
        initializer_steps: stats.initializer_steps,
        constants_materialized: stats.constants_materialized,
        runtime_variables: stats.runtime_variables,
        unsupported_initializers: stats.unsupported_initializers,
    }
}

fn initialization_diagnostics(
    runtime: &RuntimeImage,
    index: &LifecycleIndex,
) -> Vec<LifecycleDiagnostic> {
    let mut diagnostics = index.diagnostics.clone();
    diagnostics.extend(
        runtime
            .diagnostics()
            .iter()
            .map(|diagnostic| LifecycleDiagnostic {
                kind: LifecycleDiagnosticKind::UnsupportedInitializer,
                message: format!(
                    "initializer for {} requires runtime evaluation: {:?}",
                    diagnostic.variable_path, diagnostic.category
                ),
                path: Some(diagnostic.variable_path.clone()),
                source_path: Some(diagnostic.source_path.clone()),
                span: Some(diagnostic.blocker_span),
                coordinate: None,
            }),
    );
    diagnostics
}

fn plan_map_atoms(
    index: &LifecycleIndex,
    world: &WorldPlan,
    map_path: &str,
    diagnostics: &mut Vec<LifecycleDiagnostic>,
) -> Vec<PlannedAtom> {
    let mut map_atoms = Vec::new();
    for cell in world.cells() {
        let Some(template) = world.template(&cell.key) else {
            diagnostics.push(LifecycleDiagnostic {
                kind: LifecycleDiagnosticKind::MissingMapTemplate,
                message: format!("map cell key {:?} has no retained template", cell.key),
                path: None,
                source_path: Some(map_path.to_owned()),
                span: Some(cell.block_span),
                coordinate: Some(cell.coordinate),
            });
            continue;
        };
        for initializer in &template.initializers {
            let (node, category) = match initializer.resolution {
                InitializerResolution::Resolved { node, category } => (node, category),
                InitializerResolution::Unknown | InitializerResolution::NonType { .. } => {
                    diagnostics.push(LifecycleDiagnostic {
                        kind: LifecycleDiagnosticKind::UnsupportedMapInitializer,
                        message: format!(
                            "map initializer {} cannot resolve to a runtime type",
                            initializer.path
                        ),
                        path: Some(initializer.path.clone()),
                        source_path: Some(map_path.to_owned()),
                        span: Some(initializer.span),
                        coordinate: Some(cell.coordinate),
                    });
                    continue;
                }
            };
            let Some(type_index) = index.index_for_node(node) else {
                diagnostics.push(LifecycleDiagnostic {
                    kind: LifecycleDiagnosticKind::MissingTypeLifecycle,
                    message: format!(
                        "map initializer {} has no lifecycle metadata",
                        initializer.path
                    ),
                    path: Some(initializer.path.clone()),
                    source_path: Some(map_path.to_owned()),
                    span: Some(initializer.span),
                    coordinate: Some(cell.coordinate),
                });
                continue;
            };
            map_atoms.push(PlannedAtom {
                type_index,
                type_path: initializer.path.clone(),
                category,
                placement: MapPlacementContext {
                    map_path: map_path.to_owned(),
                    key: cell.key.clone(),
                    coordinate: cell.coordinate,
                    initializer_span: initializer.span,
                    block_span: cell.block_span,
                },
            });
        }
    }
    map_atoms
}

fn direct_lifecycle_procedures(
    registry: &ProcedureRegistry,
) -> BTreeMap<(NodeId, LifecycleKind), &Procedure> {
    let mut direct = BTreeMap::new();
    for procedure in registry.procedures() {
        let Some(owner) = procedure.owner_type else {
            continue;
        };
        let Some(name) = procedure.path.segments().last() else {
            continue;
        };
        for kind in LifecycleKind::ALL {
            if name == kind.procedure_name() {
                direct.insert((owner, kind), procedure);
            }
        }
    }
    direct
}

fn resolve_target(
    compilation: &Compilation,
    registry: &ProcedureRegistry,
    direct: &BTreeMap<(NodeId, LifecycleKind), &Procedure>,
    type_node: NodeId,
    kind: LifecycleKind,
) -> LifecycleResolution {
    let tree = compilation.code_tree();
    let mut current = Some(type_node);
    let mut visited = BTreeSet::new();
    while let Some(node) = current {
        if !visited.insert(node) {
            return LifecycleResolution::Unsupported(LifecycleTargetIssue {
                kind: LifecycleTargetIssueKind::MissingEffectiveTarget,
                message: "type inheritance cycle prevents lifecycle resolution".to_owned(),
                procedure_path: None,
            });
        }
        if let Some(procedure) = direct.get(&(node, kind)) {
            return resolved_procedure(compilation, registry, type_node, procedure);
        }
        current = tree.node(node).and_then(|type_node| type_node.parent_type);
    }
    LifecycleResolution::Absent
}

fn resolved_procedure(
    compilation: &Compilation,
    registry: &ProcedureRegistry,
    requested_type: NodeId,
    procedure: &Procedure,
) -> LifecycleResolution {
    let procedure_path = procedure.path.to_string();
    let Some(target_id) = procedure.effective_target else {
        return LifecycleResolution::Unsupported(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingEffectiveTarget,
            message: format!("{procedure_path} has no effective implementation"),
            procedure_path: Some(procedure_path),
        });
    };
    let Some(target) = registry.implementation(target_id) else {
        return LifecycleResolution::Unsupported(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingImplementation,
            message: format!("effective implementation for {procedure_path} is absent"),
            procedure_path: Some(procedure_path),
        });
    };
    if compilation
        .syntax(target.file_id)
        .and_then(|syntax| syntax.definitions.get(target.definition_index))
        .is_none()
    {
        return LifecycleResolution::Unsupported(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingSourceDefinition,
            message: format!("source definition for {procedure_path} is absent"),
            procedure_path: Some(procedure_path),
        });
    }
    let Some(file) = compilation.project().file(target.file_id) else {
        return LifecycleResolution::Unsupported(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingSourceDefinition,
            message: format!("source file for {procedure_path} is absent"),
            procedure_path: Some(procedure_path),
        });
    };
    let declaring_node = procedure.owner_type.unwrap_or(requested_type);
    let declaring_type = compilation
        .code_tree()
        .node(declaring_node)
        .map_or_else(|| "<unknown>".to_owned(), |node| node.path.to_string());
    LifecycleResolution::Resolved(LifecycleTarget {
        procedure: procedure.id,
        implementation: target.id,
        procedure_path,
        inherited: declaring_node != requested_type,
        declaring_type,
        source: LifecycleSource {
            file_id: target.file_id,
            path: file.relative_path.display().to_string(),
            span: compilation
                .original_span(target.file_id, target.span)
                .unwrap_or(target.span),
            ordinal: target.ordinal,
        },
    })
}

fn collect_target_diagnostics(
    type_path: &str,
    targets: &LifecycleTargets,
    diagnostics: &mut Vec<LifecycleDiagnostic>,
) {
    for kind in LifecycleKind::ALL {
        let LifecycleResolution::Unsupported(issue) = targets.get(kind) else {
            continue;
        };
        diagnostics.push(LifecycleDiagnostic {
            kind: LifecycleDiagnosticKind::UnsupportedLifecycleTarget,
            message: format!("{type_path} {kind:?}: {}", issue.message),
            path: Some(type_path.to_owned()),
            source_path: None,
            span: None,
            coordinate: None,
        });
    }
}

fn initialization_events(
    index: &LifecycleIndex,
    world_type: Option<usize>,
    atoms: &[PlannedAtom],
) -> Vec<InitializationEvent> {
    let mut events = vec![InitializationEvent::Globals];
    if let Some(type_index) = world_type
        && has_target(index, type_index, LifecycleKind::New)
    {
        events.push(InitializationEvent::Lifecycle {
            subject: EventSubject::World,
            kind: LifecycleKind::New,
            type_index,
        });
    }
    for kind in LifecycleKind::INITIALIZATION {
        for (atom_index, atom) in atoms.iter().enumerate() {
            if has_target(index, atom.type_index, kind) {
                events.push(InitializationEvent::Lifecycle {
                    subject: EventSubject::MapAtom(atom_index),
                    kind,
                    type_index: atom.type_index,
                });
            }
        }
    }
    events
}

fn has_target(index: &LifecycleIndex, type_index: usize, kind: LifecycleKind) -> bool {
    index.types.get(type_index).is_some_and(|lifecycle| {
        matches!(
            lifecycle.targets.get(kind),
            LifecycleResolution::Resolved(_)
        )
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::{Compilation, CompilerDatabase};
    use dm_map::parse;
    use dm_runtime::RuntimeImage;
    use dm_semantics::ProcedureRegistry;
    use dm_value::{FieldName, Value};
    use dm_world::{WorldCoordinate, allocate_world, build_plan};

    use super::{
        EventSubject, InitializationEvent, LifecycleIndex, LifecycleKind, LifecycleResolution,
        build_initialization_plan, execute_initialization_plan,
    };

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn compile(source: &str) -> (Self, Compilation) {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-dm-lifecycle-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&root).expect("fixture directory should be created");
            fs::write(root.join("world.dme"), "#include \"types.dm\"\n")
                .expect("environment should be written");
            fs::write(root.join("types.dm"), source).expect("source should be written");
            let compilation = CompilerDatabase::new()
                .compile(root.join("world.dme"))
                .expect("fixture should compile");
            (Self(root), compilation)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("fixture should be removed");
        }
    }

    fn index(source: &str) -> (Fixture, Compilation, RuntimeImage, LifecycleIndex) {
        let (fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let runtime =
            RuntimeImage::from_compilation(&compilation).expect("runtime image should materialize");
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        (fixture, compilation, runtime, index)
    }

    #[test]
    fn resolves_inherited_and_overridden_effective_targets() {
        let (_fixture, _compilation, _runtime, index) = index(
            "/datum/base\n\tproc/New()\n\tproc/Initialize()\n\tproc/Destroy()\n/datum/base/child\n\tInitialize()\n\tproc/LateInitialize()\n",
        );
        let child = index
            .find_path("/datum/base/child")
            .expect("child lifecycle should exist");

        let LifecycleResolution::Resolved(new_target) = &child.targets.new_target else {
            panic!("New should resolve");
        };
        assert!(new_target.inherited);
        assert_eq!(new_target.declaring_type, "/datum/base");
        let LifecycleResolution::Resolved(initialize) = &child.targets.initialize else {
            panic!("Initialize should resolve");
        };
        assert!(!initialize.inherited);
        assert_eq!(initialize.declaring_type, "/datum/base/child");
        assert!(matches!(
            child.targets.late_initialize,
            LifecycleResolution::Resolved(_)
        ));
        assert!(matches!(
            child.targets.destroy,
            LifecycleResolution::Resolved(_)
        ));
        assert!(index.diagnostics().is_empty());
    }

    #[test]
    fn plans_globals_world_and_map_lifecycles_without_execution() {
        let source = concat!(
            "/world/New()\n",
            "/atom/proc/New()\n",
            "/atom/proc/Initialize()\n",
            "/atom/proc/LateInitialize()\n",
            "/atom/proc/Destroy()\n",
            "/area/test\n",
            "/turf/test\n",
            "/obj/test\n\tInitialize()\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(5,7,2) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");

        assert_eq!(plan.map_atoms.len(), 3);
        assert_eq!(
            plan.map_atoms[0].placement.coordinate,
            WorldCoordinate { x: 5, y: 7, z: 2 }
        );
        assert_eq!(plan.map_atoms[0].placement.map_path, "test.dmm");
        assert_eq!(plan.events[0], InitializationEvent::Globals);
        assert!(matches!(
            plan.events[1],
            InitializationEvent::Lifecycle {
                subject: EventSubject::World,
                kind: LifecycleKind::New,
                ..
            }
        ));
        let lifecycle_events: Vec<_> = plan
            .events
            .iter()
            .filter_map(|event| match event {
                InitializationEvent::Lifecycle { kind, .. } => Some(*kind),
                InitializationEvent::Globals => None,
            })
            .collect();
        assert!(lifecycle_events.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(
            plan.map_lifecycle_counts(&index)[&LifecycleKind::Destroy],
            3
        );
        assert!(plan.diagnostics.is_empty());
    }

    #[test]
    fn executes_map_lifecycles_in_phase_order_without_compiling_unrelated_procs() {
        let source = concat!(
            "/world/New()\n\tsrc.stage = 5\n",
            "/atom/proc/New()\n\tsrc.stage = 10\n",
            "/atom/proc/Initialize()\n\tsrc.stage += 1\n",
            "/atom/proc/LateInitialize()\n\tsrc.stage += 100\n",
            "/area/test\n/turf/test\n/obj/test\n",
            "/proc/not_a_lifecycle_proc()\n\tspawn(1) return 0\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");

        let execution = execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("lifecycle execution should succeed");

        assert_eq!(execution.events.len(), 10);
        assert_eq!(execution.duplicate_map_events, 0);
        let kinds: Vec<_> = execution
            .events
            .iter()
            .map(|event| match event.event {
                InitializationEvent::Lifecycle { kind, .. } => kind,
                InitializationEvent::Globals => panic!("globals are not executed as a hook"),
            })
            .collect();
        assert_eq!(
            kinds,
            [
                LifecycleKind::New,
                LifecycleKind::New,
                LifecycleKind::New,
                LifecycleKind::New,
                LifecycleKind::Initialize,
                LifecycleKind::Initialize,
                LifecycleKind::Initialize,
                LifecycleKind::LateInitialize,
                LifecycleKind::LateInitialize,
                LifecycleKind::LateInitialize,
            ]
        );
        let stage = FieldName::parse("stage").expect("stage should be a field name");
        let world_id = execution.world.expect("world should be allocated");
        assert_eq!(
            runtime.heap().datum_field(world_id, &stage),
            Ok(&Value::number(5.0))
        );
        for datum in allocation.allocation_order() {
            assert_eq!(
                runtime.heap().datum_field(*datum, &stage),
                Ok(&Value::number(111.0))
            );
        }
    }
}
