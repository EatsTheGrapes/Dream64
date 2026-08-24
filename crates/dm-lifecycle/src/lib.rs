//! Deterministic, non-executing lifecycle resolution and initialization plans.

#![cfg_attr(not(test), deny(missing_docs))]

/// Versioned, self-validating storage for compiled Dream64 payloads.
pub mod artifact;
/// Loopback-only client IPC with scheduler-boundary command application.
pub mod ipc;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::sync::Arc;
use std::time::Duration;

use dm_compiler::Compilation;
use dm_core::{FileId, SourceSpan};
use dm_map::MapVariableAssignment;
use dm_object_tree::{NodeId, NodeKind};
use dm_runtime::RuntimeImage;
use dm_semantics::{Procedure, ProcedureId, ProcedureImplementationId, ProcedureRegistry};
use dm_value::{DatumId, FieldName, TypePath, Value};
use dm_vm::{
    ExecutionContext, ExecutionLimits, RuntimeError, advance_scheduler, execute_module_in_context,
};
use dm_world::{
    AtomCategory, InitializerResolution, WorldAllocation, WorldAllocationWorkKind, WorldCoordinate,
    WorldPlan, materialize_world_map_state,
};

/// Lifecycle entry points resolved for every runtime type.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LifecycleKind {
    /// Earliest world bootstrap hook, before `world/New()`.
    Genesis,
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
    const ALL: [Self; 5] = [
        Self::Genesis,
        Self::New,
        Self::Initialize,
        Self::LateInitialize,
        Self::Destroy,
    ];

    const fn procedure_name(self) -> &'static str {
        match self {
            Self::Genesis => "Genesis",
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

/// One lifecycle entry point affected by a VM compatibility issue.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleCompatibilityLocation {
    /// Lifecycle phase which reaches the selected implementation.
    pub kind: LifecycleKind,
    /// Canonical procedure path.
    pub procedure_path: String,
    /// Source definition selected for dispatch.
    pub source: LifecycleSource,
}

/// A group of equivalent VM compilation failures discovered during a sweep.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleCompatibilityIssue {
    /// Stable, coarse category derived from the compiler diagnostic.
    pub category: String,
    /// Full compiler diagnostic text.
    pub message: String,
    /// Lifecycle entry points which produced this diagnostic.
    pub locations: Vec<LifecycleCompatibilityLocation>,
}

/// Non-failing compilation audit for every lifecycle-reachable implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleCompatibilitySweep {
    /// Number of unique lifecycle implementations checked.
    pub targets: usize,
    /// Number of implementations whose complete dependency closure compiled.
    pub compatible: usize,
    /// Failures grouped by category and diagnostic message.
    pub issues: Vec<LifecycleCompatibilityIssue>,
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
    Resolved(Arc<LifecycleTarget>),
    /// A procedure was present but its target metadata was incomplete.
    Unsupported(Arc<LifecycleTargetIssue>),
}

/// All four effective lifecycle entry points for one type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleTargets {
    /// Effective pre-world bootstrap target.
    pub genesis: LifecycleResolution,
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
            LifecycleKind::Genesis => &self.genesis,
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
    /// Builds lifecycle dispatch metadata directly from compiler type nodes.
    /// This compile-only variant avoids constructing a [`RuntimeImage`] when
    /// auditing procedure compatibility before boot.
    #[must_use]
    pub fn build_compile_only(compilation: &Compilation, procedures: &ProcedureRegistry) -> Self {
        let direct = direct_lifecycle_procedures(procedures);
        let mut diagnostics = Vec::new();
        let mut types = compilation
            .code_tree()
            .nodes()
            .iter()
            .filter(|node| node.kind == NodeKind::Type)
            .map(|node| {
                let path = node.path.to_string();
                let targets = LifecycleTargets {
                    genesis: resolve_target(
                        compilation,
                        procedures,
                        &direct,
                        node.id,
                        LifecycleKind::Genesis,
                    ),
                    new_target: resolve_target(
                        compilation,
                        procedures,
                        &direct,
                        node.id,
                        LifecycleKind::New,
                    ),
                    initialize: resolve_target(
                        compilation,
                        procedures,
                        &direct,
                        node.id,
                        LifecycleKind::Initialize,
                    ),
                    late_initialize: resolve_target(
                        compilation,
                        procedures,
                        &direct,
                        node.id,
                        LifecycleKind::LateInitialize,
                    ),
                    destroy: resolve_target(
                        compilation,
                        procedures,
                        &direct,
                        node.id,
                        LifecycleKind::Destroy,
                    ),
                };
                collect_target_diagnostics(&path, &targets, &mut diagnostics);
                TypeLifecycle {
                    node: node.id,
                    path,
                    parent: node
                        .parent_type
                        .and_then(|parent| compilation.code_tree().node(parent))
                        .map(|parent| parent.path.to_string()),
                    targets,
                }
            })
            .collect::<Vec<_>>();
        types.sort_by(|left, right| left.path.cmp(&right.path));
        share_resolved_lifecycle_targets(&mut types);
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
                genesis: resolve_target(
                    compilation,
                    procedures,
                    &direct,
                    node,
                    LifecycleKind::Genesis,
                ),
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
        share_resolved_lifecycle_targets(&mut types);
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
    pub map_path: Arc<str>,
    /// Cell key that supplied the initializer template.
    pub key: Arc<str>,
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
    pub type_path: Arc<str>,
    /// World atom category inferred by the map planner.
    pub category: AtomCategory,
    /// Ordered, lossless map variable assignments for this atom placement.
    ///
    /// The values remain unevaluated until map initialization execution. Each
    /// assignment retains its target and value source spans, along with its raw
    /// source text, through [`MapVariableAssignment`].
    pub variables: Arc<[MapVariableAssignment]>,
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
    /// Globals-first, world Genesis, compiled-map construction, world New, then
    /// either engine-managed or project-managed atom initialization.
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
    /// Total successfully executed hooks, including production boots that
    /// release the detailed per-event audit trail after initialization.
    pub executed_events: usize,
    /// Successfully executed hooks grouped by lifecycle kind.
    pub executed_event_counts: BTreeMap<LifecycleKind, usize>,
    /// Repeated map placements sharing an already initialized datum.
    pub duplicate_map_events: usize,
    /// Deterministic scheduler work completed after lifecycle initialization.
    pub scheduler: SchedulerDrain,
}

/// Safety bounds for the post-initialization deterministic scheduler drain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerDrainLimits {
    /// Maximum scheduler ticks advanced from the lifecycle completion tick.
    pub max_ticks: u64,
    /// Maximum dispatch rounds, including zero-delay rescheduling rounds.
    pub max_rounds: usize,
}

/// Adaptive instruction budget for latency-sensitive persistent host slices.
///
/// The VM remains single-owner. This controller only changes how frequently
/// execution returns to the host so sockets, timers, and completed immutable
/// worker jobs can be serviced.
#[derive(Clone, Debug)]
pub struct HostSliceBudget {
    current_steps: u64,
    minimum_steps: u64,
    maximum_steps: u64,
    target: Duration,
}

impl HostSliceBudget {
    /// Creates a bounded controller, clamping the initial budget into range.
    #[must_use]
    pub fn new(
        initial_steps: u64,
        minimum_steps: u64,
        maximum_steps: u64,
        target: Duration,
    ) -> Self {
        let minimum_steps = minimum_steps.max(1);
        let maximum_steps = maximum_steps.max(minimum_steps);
        Self {
            current_steps: initial_steps.clamp(minimum_steps, maximum_steps),
            minimum_steps,
            maximum_steps,
            target: target.max(Duration::from_micros(1)),
        }
    }

    /// Instruction ceiling for the next persistent scheduler round.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.current_steps
    }

    /// Records VM wall time and adjusts the following instruction ceiling.
    ///
    /// An over-target slice halves immediately. Sustained slices below half
    /// the target recover gradually, avoiding oscillation around the target.
    pub fn observe(&mut self, elapsed: Duration) {
        if elapsed > self.target {
            self.current_steps = (self.current_steps / 2).max(self.minimum_steps);
        } else if elapsed <= self.target / 2 {
            let growth = (self.current_steps / 4).max(1);
            self.current_steps = self
                .current_steps
                .saturating_add(growth)
                .min(self.maximum_steps);
        }
    }
}

/// Explicit runtime state which proves a persistent server finished startup.
///
/// The probe begins at a global and may follow datum fields before comparing
/// the resulting value. This lets a codebase expose its own authoritative
/// readiness marker instead of mistaking a scheduler budget for success.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadlessReadinessProbe {
    /// Owner-qualified VM slot when the readiness marker is a type static.
    pub qualified_storage: Option<FieldName>,
    /// Runtime global containing the marker or its root datum.
    pub global: FieldName,
    /// Datum fields followed from the global value, in order.
    pub fields: Vec<FieldName>,
    /// Value which denotes completed startup.
    pub expected: Value,
}

/// Lifecycle bytecode linked before runtime/world materialization so boot does
/// not overlap its closure/spec peak with the resident runtime heap.
pub struct PrecompiledLifecycle {
    executable: dm_semantics::ExecutableProcedures,
    persistent_state: Option<dm_vm::ExecutionState>,
    targets: usize,
    reachable_bodies: usize,
    closure: dm_semantics::ProcedureClosureStats,
}

impl PrecompiledLifecycle {
    /// Complete executable module used to validate/restored scheduler frames.
    #[doc(hidden)]
    #[must_use]
    pub fn module(&self) -> &dm_vm::Module {
        self.executable.module()
    }

    /// Installs a validated ready-world state loaded from the startup cache.
    #[doc(hidden)]
    pub fn install_persistent_state(&mut self, state: dm_vm::ExecutionState) {
        self.persistent_state = Some(state);
    }

    /// Returns the complete linked module so runtime initializer expressions
    /// can be appended without relinking the project inventory.
    #[doc(hidden)]
    pub fn module_mut_for_runtime_initializers(&mut self) -> &mut dm_vm::Module {
        self.executable.module_mut()
    }

    /// Exact effective lifecycle roots selected from the world/map plan.
    #[must_use]
    pub const fn targets(&self) -> usize {
        self.targets
    }

    /// Reachable implementation bodies retained by symbolic linking.
    #[must_use]
    pub const fn reachable_bodies(&self) -> usize {
        self.reachable_bodies
    }

    /// Deterministic dependency-closure counters.
    #[must_use]
    pub const fn closure_stats(&self) -> &dm_semantics::ProcedureClosureStats {
        &self.closure
    }

    /// Symbolically linked procedure bodies, including deferred bodies.
    #[must_use]
    pub fn module_procedures(&self) -> usize {
        self.executable.stats().procedures
    }

    /// Deferred bodies not materialized during precompile.
    #[must_use]
    pub fn deferred_procedures(&self) -> usize {
        self.executable.module().deferred_procedure_count()
    }

    /// Host duration of one current BYOND world tick for persistent pacing.
    #[must_use]
    pub fn persistent_tick_duration(&self) -> std::time::Duration {
        let tick_lag = self
            .persistent_state
            .as_ref()
            .and_then(|state| {
                let world = state.global(&FieldName::parse("world").ok()?)?;
                let Value::Datum(world) = world else {
                    return None;
                };
                state
                    .heap()
                    .datum_field(*world, &FieldName::parse("tick_lag").ok()?)
                    .ok()?
                    .as_number()
            })
            .filter(|tick_lag| tick_lag.is_finite() && *tick_lag > 0.0)
            .unwrap_or(1.0);
        std::time::Duration::from_secs_f64(f64::from(tick_lag) / 10.0)
    }

    /// Returns the live VM state retained by a completed headless boot.
    ///
    /// Hosts use this at the persistent-scheduler boundary to service local
    /// client sessions without duplicating world state in a second runtime.
    #[must_use]
    pub fn persistent_state_mut(&mut self) -> Option<&mut dm_vm::ExecutionState> {
        self.persistent_state.as_mut()
    }

    /// Returns bounded suspended-DM telemetry for controlled host shutdown.
    #[must_use]
    pub fn bounded_scheduler_progress(&self) -> Vec<String> {
        self.persistent_state
            .as_ref()
            .map_or_else(Vec::new, |state| {
                state.bounded_scheduler_progress(self.executable.module())
            })
    }

    /// Installs a loopback guest in the persistent world and queues the
    /// project's effective `/client/New()` hook on the world scheduler.
    ///
    /// # Errors
    ///
    /// Returns an error when persistent startup is incomplete or the guest
    /// cannot be created and scheduled.
    pub fn connect_local_guest(&mut self) -> Result<dm_vm::LocalClientState, String> {
        let state = self
            .persistent_state
            .as_mut()
            .ok_or_else(|| "persistent world is not ready for clients".to_owned())?;
        state.connect_local_guest(self.executable.module())
    }

    /// Queues one command through the connected client's effective verb set.
    ///
    /// # Errors
    ///
    /// Returns an error when startup is incomplete or command resolution fails.
    pub fn queue_local_client_command(
        &mut self,
        client: DatumId,
        command: &str,
    ) -> Result<(), String> {
        let state = self
            .persistent_state
            .as_mut()
            .ok_or_else(|| "persistent world is not ready for clients".to_owned())?;
        state.queue_local_client_command(self.executable.module(), client, command)
    }

    /// Queues one browser topic against the connected client's `Topic()`.
    pub fn queue_local_browser_topic(
        &mut self,
        client: DatumId,
        topic: &str,
    ) -> Result<(), String> {
        let state = self
            .persistent_state
            .as_mut()
            .ok_or_else(|| "persistent world is not ready for clients".to_owned())?;
        state.queue_local_browser_topic(self.executable.module(), client, topic)
    }

    /// Queues one pointer event for a session-owned screen object.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_local_screen_pointer(
        &mut self,
        client: DatumId,
        index: u32,
        generation: u32,
        event: dm_vm::LocalScreenPointerEvent,
        location: &str,
        params: &str,
    ) -> Result<(), String> {
        let state = self
            .persistent_state
            .as_mut()
            .ok_or_else(|| "persistent world is not ready for clients".to_owned())?;
        state.queue_local_screen_pointer(
            self.executable.module(),
            client,
            index,
            generation,
            event,
            location,
            params,
        )
    }

    /// Queues one click for an atom rendered in a session-visible map cell.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_local_map_pointer(
        &mut self,
        client: DatumId,
        index: u32,
        generation: u32,
        x: i32,
        y: i32,
        z: i32,
        control: &str,
        params: &str,
    ) -> Result<(), String> {
        let state = self
            .persistent_state
            .as_mut()
            .ok_or_else(|| "persistent world is not ready for clients".to_owned())?;
        state.queue_local_map_pointer(
            self.executable.module(),
            client,
            index,
            generation,
            x,
            y,
            z,
            control,
            params,
        )
    }
}

/// Selects and symbolically links the exact world/map lifecycle roots without
/// constructing a runtime image or allocating map atoms.
///
/// # Errors
///
/// Returns a lowering error when a required eager lifecycle body cannot be
/// represented by the VM.
pub fn precompile_lifecycle_for_world(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    world: &WorldPlan,
) -> Result<PrecompiledLifecycle, dm_vm::CompileError> {
    let targets = lifecycle_targets_for_world(index, world);
    let executable = procedures
        .compile_vm_all_symbolic_with_eager_roots(compilation, targets.iter().copied())?;
    Ok(precompiled_lifecycle_from_executable(
        compilation,
        procedures,
        &targets,
        executable,
    ))
}

/// Selects the exact world/map lifecycle roots and closure while reusing an
/// already linked executable module.
///
/// This is the artifact-backed counterpart to
/// [`precompile_lifecycle_for_world`]. It performs no linking or lowering;
/// the caller-supplied executable is moved directly into the boot state.
#[must_use]
pub fn precompile_lifecycle_for_world_with_executable(
    _compilation: &Compilation,
    _procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    world: &WorldPlan,
    executable: dm_semantics::ExecutableProcedures,
) -> PrecompiledLifecycle {
    let targets = lifecycle_targets_for_world(index, world);
    // An artifact-backed executable already contains the complete linked
    // procedure inventory. Rebuilding the compiler's transitive dependency
    // closure here cannot alter executable contents or runtime dispatch; it
    // previously spent tens of seconds solely to populate diagnostic counters.
    // Keep exact target cardinality and report the resident executable body
    // count while leaving closure counters empty on this runtime-only path.
    let reachable_bodies = executable.stats().procedures;
    PrecompiledLifecycle {
        executable,
        persistent_state: None,
        targets: targets.len(),
        reachable_bodies,
        closure: dm_semantics::ProcedureClosureStats::default(),
    }
}

fn lifecycle_targets_for_world(
    index: &LifecycleIndex,
    world: &WorldPlan,
) -> BTreeSet<ProcedureImplementationId> {
    let mut targets = BTreeSet::new();
    if let Some(world_type) = index.find_path("/world") {
        for kind in [LifecycleKind::Genesis, LifecycleKind::New] {
            if let LifecycleResolution::Resolved(target) = world_type.targets.get(kind) {
                targets.insert(target.implementation);
            }
        }
    }
    let map_kinds: &[LifecycleKind] = if project_manages_atom_initialization(index) {
        &[LifecycleKind::New]
    } else {
        &[
            LifecycleKind::New,
            LifecycleKind::Initialize,
            LifecycleKind::LateInitialize,
        ]
    };
    for node in world
        .templates()
        .values()
        .flat_map(|template| &template.initializers)
        .filter_map(|initializer| match initializer.resolution {
            InitializerResolution::Resolved { node, .. } => Some(node),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
    {
        let Some(lifecycle) = index.find_node(node) else {
            continue;
        };
        for &kind in map_kinds {
            if let LifecycleResolution::Resolved(target) = lifecycle.targets.get(kind) {
                targets.insert(target.implementation);
            }
        }
    }
    targets
}

fn precompiled_lifecycle_from_executable(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    targets: &BTreeSet<ProcedureImplementationId>,
    executable: dm_semantics::ExecutableProcedures,
) -> PrecompiledLifecycle {
    let target_count = targets.len();
    let (reachable, closure) =
        procedures.implementation_closure_with_stats(compilation, targets.iter().copied());
    PrecompiledLifecycle {
        executable,
        persistent_state: None,
        targets: target_count,
        reachable_bodies: reachable.len(),
        closure,
    }
}

impl Default for SchedulerDrainLimits {
    fn default() -> Self {
        Self {
            max_ticks: 10_000,
            max_rounds: 10_000,
        }
    }
}

/// Honest reason the bounded post-initialization scheduler drain stopped.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SchedulerDrainTermination {
    /// No scheduled work remains.
    #[default]
    StableIdle,
    /// The configured codebase-owned readiness marker was observed while
    /// persistent scheduled work remained.
    HeadlessReady,
    /// Work remains beyond the configured tick budget.
    TickLimit,
    /// Work remains after the configured dispatch-round budget.
    RoundLimit,
}

/// Deterministic post-initialization scheduler summary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SchedulerDrain {
    /// Scheduler tick at which draining stopped.
    pub final_tick: u64,
    /// Number of dispatch rounds performed.
    pub rounds: usize,
    /// Number of tasks which ran to completion.
    pub completed_tasks: usize,
    /// Number of persistent scheduled threads terminated by an isolated
    /// runtime error during this drain.
    ///
    /// Startup drains continue by default and report any startup-thread
    /// failure in this field.
    pub failed_tasks: usize,
    /// Tasks still pending when draining stopped.
    pub pending_tasks: usize,
    /// Why draining stopped.
    pub termination: SchedulerDrainTermination,
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
    /// A deferred map field expression could not be applied before lifecycle hooks.
    MapExpression {
        /// Index into [`InitializationPlan::map_atoms`].
        atom_index: usize,
        /// Target field name from the map assignment.
        field: String,
        /// Original map source range for the full assignment.
        span: SourceSpan,
        /// Original lowering or execution failure.
        error: Box<dm_runtime::RuntimeImageError>,
    },
    /// The runtime image cannot allocate the singleton `/world` datum.
    WorldAllocation(dm_runtime::RuntimeImageError),
    /// Map-derived dimensions or contents could not be applied to `/world`.
    WorldMapState(dm_world::WorldAllocationError),
    /// VM execution failed with its original source-mapped call stack.
    Runtime {
        /// Event being executed.
        event: InitializationEvent,
        /// Source-selected lifecycle target.
        target: Box<LifecycleTarget>,
        /// Original VM failure.
        error: Box<RuntimeError>,
    },
    /// Diagnostic execution completed all independent map lifecycle hooks and
    /// collected one or more unique runtime failures.
    AuditFailures {
        /// Number of unique procedure/error groups printed during the audit.
        failures: usize,
    },
    /// A spawned or sleeping task failed during the scheduler drain.
    Scheduler(RuntimeError),
}

/// Failure while allocating a datum and dispatching its effective `New` procedure.
#[derive(Debug)]
pub enum ConstructionError {
    /// Runtime defaults could not be materialized for the requested type.
    Allocation(dm_runtime::RuntimeImageError),
    /// The selected constructor closure could not be lowered.
    Compile(dm_vm::CompileError),
    /// Lifecycle metadata for the requested type or constructor is incomplete.
    MissingTarget(String),
    /// Constructor metadata was resolved but omitted from the compiled module.
    MissingVmTarget(String),
    /// Constructor execution failed after allocation; the new datum was destroyed.
    Runtime(RuntimeError),
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation(error) => write!(formatter, "datum allocation failed: {error}"),
            Self::Compile(error) => write!(formatter, "constructor compilation failed: {error}"),
            Self::MissingTarget(path) => {
                write!(formatter, "constructor target is missing for {path}")
            }
            Self::MissingVmTarget(path) => {
                write!(formatter, "constructor VM target is missing for {path}")
            }
            Self::Runtime(error) => write!(formatter, "constructor execution failed: {error}"),
        }
    }
}

impl std::error::Error for ConstructionError {}

/// Failure while dispatching cleanup for a live datum.
#[derive(Debug)]
pub enum DeletionError {
    /// The datum is stale or its runtime type is unavailable.
    Datum(dm_value::ValueError),
    /// Lifecycle metadata for the datum type is incomplete.
    MissingTarget(String),
    /// The selected cleanup closure could not be lowered.
    Compile(dm_vm::CompileError),
    /// Cleanup metadata was resolved but omitted from the VM module.
    MissingVmTarget(String),
    /// Cleanup execution failed. The datum was still invalidated.
    Runtime(RuntimeError),
}

impl std::fmt::Display for DeletionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Datum(error) => write!(formatter, "datum deletion failed: {error}"),
            Self::MissingTarget(path) => write!(formatter, "cleanup target is missing for {path}"),
            Self::Compile(error) => write!(formatter, "cleanup compilation failed: {error}"),
            Self::MissingVmTarget(path) => {
                write!(formatter, "cleanup VM target is missing for {path}")
            }
            Self::Runtime(error) => write!(formatter, "cleanup execution failed: {error}"),
        }
    }
}

impl std::error::Error for DeletionError {}

/// Allocates one datum, applies its inherited/default field layers, and invokes
/// the effective `New` implementation with the supplied arguments.
///
/// A type without a user-defined `New` succeeds after default materialization.
/// Constructor closures include exact `..()` parent targets. If execution
/// fails, the newly allocated datum is removed from the shared heap before the
/// error is returned.
pub fn construct_datum(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    runtime: &mut RuntimeImage,
    type_path: &TypePath,
    arguments: &[Value],
) -> Result<DatumId, ConstructionError> {
    let index = LifecycleIndex::build(compilation, procedures, runtime);
    let resolution = index
        .find_path(type_path.as_str())
        .ok_or_else(|| ConstructionError::MissingTarget(type_path.to_string()))?
        .targets
        .get(LifecycleKind::New)
        .clone();
    let target = match resolution {
        LifecycleResolution::Absent => {
            return runtime
                .allocate_datum(type_path)
                .map_err(ConstructionError::Allocation);
        }
        LifecycleResolution::Resolved(target) => target,
        LifecycleResolution::Unsupported(_) => {
            return Err(ConstructionError::MissingTarget(type_path.to_string()));
        }
    };
    let executable = procedures
        .compile_vm_implementations(compilation, [target.implementation])
        .map_err(ConstructionError::Compile)?;
    let entry = executable
        .implementation(target.implementation)
        .ok_or_else(|| ConstructionError::MissingVmTarget(target.procedure_path.clone()))?;
    let datum = runtime
        .allocate_datum(type_path)
        .map_err(ConstructionError::Allocation)?;
    let mut state = runtime.take_execution_state();
    let result = execute_module_in_context(
        executable.module(),
        entry,
        arguments,
        &mut state,
        &ExecutionContext::new(Value::Datum(datum), Value::Null),
    );
    if result.is_err() {
        let _ = state.heap_mut().destroy_datum(datum);
    }
    runtime.restore_execution_state(state);
    result.map(|_| datum).map_err(ConstructionError::Runtime)
}

/// Dispatches the effective qdel-compatible `Destroy` chain once and then
/// invalidates the datum handle. Cleanup failure never leaves the datum live.
/// A cleanup body that deletes `src` reentrantly is tolerated; the final stale
/// destroy is treated as already complete.
pub fn delete_datum(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    runtime: &mut RuntimeImage,
    datum: DatumId,
) -> Result<(), DeletionError> {
    let type_path = runtime
        .heap()
        .datum(datum)
        .map_err(DeletionError::Datum)?
        .type_path()
        .clone();
    let index = LifecycleIndex::build(compilation, procedures, runtime);
    let resolution = index
        .find_path(type_path.as_str())
        .ok_or_else(|| DeletionError::MissingTarget(type_path.to_string()))?
        .targets
        .get(LifecycleKind::Destroy)
        .clone();
    let target = match resolution {
        LifecycleResolution::Absent => {
            runtime
                .heap_mut()
                .destroy_datum(datum)
                .map_err(DeletionError::Datum)?;
            return Ok(());
        }
        LifecycleResolution::Resolved(target) => target,
        LifecycleResolution::Unsupported(_) => {
            return Err(DeletionError::MissingTarget(type_path.to_string()));
        }
    };
    let executable = procedures
        .compile_vm_implementations(compilation, [target.implementation])
        .map_err(DeletionError::Compile)?;
    let entry = executable
        .implementation(target.implementation)
        .ok_or_else(|| DeletionError::MissingVmTarget(target.procedure_path.clone()))?;
    let mut state = runtime.take_execution_state();
    let result = execute_module_in_context(
        executable.module(),
        entry,
        &[],
        &mut state,
        &ExecutionContext::new(Value::Datum(datum), Value::Null),
    );
    let _ = state.heap_mut().destroy_datum(datum);
    runtime.restore_execution_state(state);
    result.map(|_| ()).map_err(DeletionError::Runtime)
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
            Self::MapExpression {
                atom_index,
                field,
                span,
                error,
            } => write!(
                formatter,
                "map expression for atom {atom_index} field {field} at {}..{} failed: {error}",
                span.start, span.end
            ),
            Self::WorldAllocation(error) => write!(formatter, "world allocation failed: {error}"),
            Self::WorldMapState(error) => {
                write!(formatter, "world map state materialization failed: {error}")
            }
            Self::Runtime { target, error, .. } => {
                write!(
                    formatter,
                    "lifecycle {} failed: {error}",
                    target.procedure_path
                )
            }
            Self::AuditFailures { failures } => write!(
                formatter,
                "lifecycle audit collected {failures} unique runtime failure groups"
            ),
            Self::Scheduler(error) => write!(formatter, "scheduled startup task failed: {error}"),
        }
    }
}

impl std::error::Error for InitializationExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compile(error) => Some(error),
            Self::WorldPath(error) => Some(error),
            Self::WorldAllocation(error) => Some(error),
            Self::WorldMapState(error) => Some(error),
            Self::MapExpression { error, .. } => Some(error),
            Self::Runtime { error, .. } => Some(error),
            Self::Scheduler(error) => Some(error),
            Self::MissingMapDatum { .. }
            | Self::MissingTarget { .. }
            | Self::MissingVmTarget { .. }
            | Self::MissingWorldDatum
            | Self::AuditFailures { .. } => None,
        }
    }
}

/// Executes world `Genesis`/`New`, then mapped `New`, `Initialize`, and
/// `LateInitialize` hooks.
///
/// The caller first builds an [`InitializationPlan`] and materializes the same
/// [`WorldPlan`] with [`dm_world::allocate_world`]. Hooks use each live datum as
/// `src`, execute in plan order, and share one mutable VM heap. Repeated map
/// placements referring to one shared area datum run each hook once.
///
/// # Panics
///
/// Panics only if Dream64's hard-coded `world` built-in identifier stops being
/// a valid DM field name, which would violate an internal engine invariant.
///
/// # Errors
///
/// Returns a source-aware error when a planned target cannot be compiled,
/// bound to an allocation, or executed.
#[allow(clippy::too_many_lines)]
pub fn execute_initialization_plan(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
) -> Result<InitializationExecution, InitializationExecutionError> {
    execute_initialization_plan_with_scheduler_limits(
        compilation,
        procedures,
        index,
        plan,
        allocation,
        runtime,
        SchedulerDrainLimits::default(),
    )
}

/// Executes lifecycle initialization and then drains deterministic scheduled
/// startup work within explicit safety bounds.
///
/// # Errors
///
/// Returns the same source-aware lifecycle failures as
/// [`execute_initialization_plan`], or a scheduler runtime failure.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn execute_initialization_plan_with_scheduler_limits(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
    scheduler_limits: SchedulerDrainLimits,
) -> Result<InitializationExecution, InitializationExecutionError> {
    execute_initialization_plan_with_scheduler_policy(
        compilation,
        procedures,
        index,
        plan,
        allocation,
        runtime,
        scheduler_limits,
        None,
    )
}

/// Executes lifecycle initialization with an optional codebase-owned
/// persistent-server readiness marker.
///
/// # Errors
///
/// Returns the same source-aware lifecycle and scheduler failures as
/// [`execute_initialization_plan_with_scheduler_limits`].
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn execute_initialization_plan_with_scheduler_policy(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
    scheduler_limits: SchedulerDrainLimits,
    readiness: Option<&HeadlessReadinessProbe>,
) -> Result<InitializationExecution, InitializationExecutionError> {
    eprintln!("boot-progress: selecting lifecycle targets");
    let targets = plan
        .events
        .iter()
        .filter_map(|event| event_target(*event, index).map(|target| target.implementation))
        .collect::<BTreeSet<_>>();
    eprintln!(
        "boot-progress: selected lifecycle targets={}",
        targets.len()
    );
    let (reachable, closure_stats) =
        procedures.implementation_closure_with_stats(compilation, targets.iter().copied());
    eprintln!(
        "boot-progress: lifecycle closure bodies={} static_selectors={} dynamic_selectors={} dynamic_candidates={}",
        reachable.len(),
        closure_stats.static_selectors_resolved,
        closure_stats.dynamic_selectors_resolved,
        closure_stats.dynamic_candidates_considered,
    );
    eprintln!("boot-progress: compiling lifecycle targets");
    let mut executable = procedures
        .compile_vm_implementations_symbolic_dynamic(compilation, targets)
        .map_err(InitializationExecutionError::Compile)?;
    execute_initialization_plan_with_executable(
        index,
        plan,
        allocation,
        runtime,
        scheduler_limits,
        readiness,
        &mut executable,
        None,
        false,
        false,
        None,
    )
}

/// Executes a plan using lifecycle bytecode linked before runtime/world
/// materialization.
#[allow(clippy::too_many_arguments)]
pub fn execute_initialization_plan_with_precompiled(
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
    scheduler_limits: SchedulerDrainLimits,
    readiness: Option<&HeadlessReadinessProbe>,
    precompiled: &mut PrecompiledLifecycle,
) -> Result<InitializationExecution, InitializationExecutionError> {
    eprintln!(
        "boot-progress: using precompiled lifecycle targets={} bodies={} procedures={} deferred={}",
        precompiled.targets,
        precompiled.reachable_bodies,
        precompiled.module_procedures(),
        precompiled.deferred_procedures(),
    );
    if readiness.is_some() {
        execute_initialization_plan_with_executable(
            index,
            plan,
            allocation,
            runtime,
            scheduler_limits,
            readiness,
            &mut precompiled.executable,
            Some(&mut precompiled.persistent_state),
            false,
            false,
            None,
        )
    } else {
        execute_initialization_plan_with_executable(
            index,
            plan,
            allocation,
            runtime,
            scheduler_limits,
            None,
            &mut precompiled.executable,
            None,
            false,
            false,
            None,
        )
    }
}

/// Executes production boot using precompiled lifecycle bytecode and releases
/// cold host metadata once dynamic map overrides have entered the live VM.
///
/// # Errors
///
/// Returns the same failures as [`execute_initialization_plan_with_precompiled`].
/// The supplied runtime image is intentionally no longer reusable afterward.
#[allow(clippy::too_many_arguments)]
pub fn execute_boot_initialization_plan_with_precompiled(
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
    scheduler_limits: SchedulerDrainLimits,
    readiness: Option<&HeadlessReadinessProbe>,
    precompiled: &mut PrecompiledLifecycle,
) -> Result<InitializationExecution, InitializationExecutionError> {
    eprintln!(
        "boot-progress: using precompiled lifecycle targets={} bodies={} procedures={} deferred={}",
        precompiled.targets,
        precompiled.reachable_bodies,
        precompiled.module_procedures(),
        precompiled.deferred_procedures(),
    );
    execute_initialization_plan_with_executable(
        index,
        plan,
        allocation,
        runtime,
        scheduler_limits,
        readiness,
        &mut precompiled.executable,
        readiness.map(|_| &mut precompiled.persistent_state),
        false,
        true,
        None,
    )
}

/// Executes production boot while servicing a local client endpoint during
/// the startup scheduler drain. This allows clients to attach after
/// `world/New()` has returned while Master continues bringing subsystems to
/// authoritative readiness.
///
/// # Errors
///
/// Returns the same failures as [`execute_boot_initialization_plan_with_precompiled`].
#[allow(clippy::too_many_arguments)]
pub fn execute_boot_initialization_plan_with_precompiled_and_startup_service(
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
    scheduler_limits: SchedulerDrainLimits,
    readiness: Option<&HeadlessReadinessProbe>,
    precompiled: &mut PrecompiledLifecycle,
    startup_service: &mut dyn FnMut(
        &dm_semantics::ExecutableProcedures,
        &mut dm_vm::ExecutionState,
    ),
) -> Result<InitializationExecution, InitializationExecutionError> {
    eprintln!(
        "boot-progress: using precompiled lifecycle targets={} bodies={} procedures={} deferred={}",
        precompiled.targets,
        precompiled.reachable_bodies,
        precompiled.module_procedures(),
        precompiled.deferred_procedures(),
    );
    execute_initialization_plan_with_executable(
        index,
        plan,
        allocation,
        runtime,
        scheduler_limits,
        readiness,
        &mut precompiled.executable,
        readiness.map(|_| &mut precompiled.persistent_state),
        false,
        true,
        Some(startup_service),
    )
}

/// Executes all independent mapped lifecycle hooks while collecting unique
/// runtime failures instead of stopping at the first failed map datum.
///
/// World/Genesis failures remain fail-fast because every mapped object depends
/// on that shared state. A failed datum is skipped for its remaining lifecycle
/// phases so one root problem does not create duplicate Initialize/Late errors.
#[allow(clippy::too_many_arguments)]
pub fn audit_initialization_plan_with_precompiled(
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
    scheduler_limits: SchedulerDrainLimits,
    precompiled: &mut PrecompiledLifecycle,
) -> Result<InitializationExecution, InitializationExecutionError> {
    eprintln!(
        "boot-progress: auditing precompiled lifecycle targets={} bodies={} procedures={} deferred={}",
        precompiled.targets,
        precompiled.reachable_bodies,
        precompiled.module_procedures(),
        precompiled.deferred_procedures(),
    );
    execute_initialization_plan_with_executable(
        index,
        plan,
        allocation,
        runtime,
        scheduler_limits,
        None,
        &mut precompiled.executable,
        None,
        true,
        false,
        None,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn execute_initialization_plan_with_executable(
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    runtime: &mut RuntimeImage,
    scheduler_limits: SchedulerDrainLimits,
    readiness: Option<&HeadlessReadinessProbe>,
    executable: &mut dm_semantics::ExecutableProcedures,
    persistent_state: Option<&mut Option<dm_vm::ExecutionState>>,
    collect_runtime_errors: bool,
    release_runtime_metadata: bool,
    startup_service: Option<
        &mut dyn FnMut(&dm_semantics::ExecutableProcedures, &mut dm_vm::ExecutionState),
    >,
) -> Result<InitializationExecution, InitializationExecutionError> {
    eprintln!("boot-progress: selecting lifecycle targets");
    let bindings = map_datum_bindings(plan, allocation, runtime);
    eprintln!(
        "boot-progress: compiled lifecycle targets deferred={} materialized={}",
        executable.module().deferred_procedure_count(),
        executable.module().materialized_deferred_procedure_count(),
    );
    let world = if plan.world_type.is_some() {
        let world = if let Some(world) = runtime.canonical_world() {
            world
        } else {
            runtime
                .allocate_datum(
                    &TypePath::parse("/world").map_err(InitializationExecutionError::WorldPath)?,
                )
                .map_err(InitializationExecutionError::WorldAllocation)?
        };
        materialize_world_map_state(allocation, runtime, world)
            .map_err(InitializationExecutionError::WorldMapState)?;
        Some(world)
    } else {
        None
    };

    let mut state = runtime.take_execution_state();
    if let Some(world) = world {
        state.set_global(
            FieldName::parse("world").expect("built-in world global name is valid"),
            Value::Datum(world),
        );
    }
    let execution = (|| {
        eprintln!("boot-progress: applying dynamic map overrides");
        apply_dynamic_map_overrides(
            plan,
            allocation,
            &bindings,
            runtime,
            &mut state,
            executable.module_mut(),
        )?;
        eprintln!("boot-progress: applied dynamic map overrides");
        if release_runtime_metadata {
            let released = runtime.release_transferred_metadata();
            eprintln!(
                "boot-progress: released transferred metadata variables={} types={} initializer_candidates={}",
                released.variables, released.types, released.initializer_candidates,
            );
        }
        let mut result = InitializationExecution {
            world,
            ..InitializationExecution::default()
        };
        let mut seen = BTreeSet::new();
        let mut initialized_during_new = BTreeSet::new();
        let mut failed_datums = BTreeSet::new();
        let mut audit_failures = BTreeSet::new();
        let total_lifecycle_events = plan
            .events
            .iter()
            .filter(|event| matches!(event, InitializationEvent::Lifecycle { .. }))
            .count();
        let mut lifecycle_event = 0usize;
        eprintln!("boot-progress: executing lifecycle events total={total_lifecycle_events}");
        for event in &plan.events {
            let InitializationEvent::Lifecycle { subject, .. } = *event else {
                continue;
            };
            lifecycle_event += 1;
            if lifecycle_event == 1 || lifecycle_event % 10_000 == 0 {
                eprintln!(
                    "boot-progress: lifecycle event {lifecycle_event}/{total_lifecycle_events} phase={:?}",
                    event_kind(*event),
                );
            }
            let datum = match subject {
                EventSubject::World => {
                    world.ok_or(InitializationExecutionError::MissingWorldDatum)?
                }
                EventSubject::MapAtom(atom_index) => bindings
                    .get(atom_index)
                    .and_then(|datum| *datum)
                    .ok_or_else(|| InitializationExecutionError::MissingMapDatum {
                        atom_index,
                        path: plan.map_atoms[atom_index].type_path.to_string(),
                    })?,
                EventSubject::Globals => continue,
            };
            if matches!(subject, EventSubject::MapAtom(_))
                && !seen.insert((datum, event_kind(*event)))
            {
                result.duplicate_map_events += 1;
                continue;
            }
            if failed_datums.contains(&datum) {
                continue;
            }
            if matches!(subject, EventSubject::MapAtom(_))
                && matches!(
                    event_kind(*event),
                    LifecycleKind::Initialize | LifecycleKind::LateInitialize
                )
                && initialized_during_new.contains(&datum)
            {
                // Monk/tg's INITIALIZE_IMMEDIATE macro temporarily enables
                // SSatoms from inside New(), which runs Initialize and queues
                // any LateInitialize itself. Do not synthesize those hooks a
                // second time for datums carrying INITIALIZED_1 afterward.
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
            let arguments = if matches!(subject, EventSubject::MapAtom(_))
                && event_kind(*event) == LifecycleKind::New
            {
                vec![
                    state
                        .heap()
                        .datum_field(
                            datum,
                            &FieldName::parse("loc")
                                .expect("built-in atom loc field name is valid"),
                        )
                        .cloned()
                        .unwrap_or(Value::Null),
                ]
            } else {
                Vec::new()
            };
            let value = match execute_module_in_context(
                executable.module(),
                entry,
                &arguments,
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ) {
                Ok(value) => value,
                Err(error)
                    if collect_runtime_errors && matches!(subject, EventSubject::MapAtom(_)) =>
                {
                    failed_datums.insert(datum);
                    let key = format!("{}: {error}", target.procedure_path);
                    if audit_failures.insert(key.clone()) {
                        eprintln!(
                            "boot-audit-runtime-error: group={} event={:?} datum={:?} {}",
                            audit_failures.len(),
                            event_kind(*event),
                            datum,
                            key,
                        );
                    }
                    continue;
                }
                Err(error) => {
                    return Err(InitializationExecutionError::Runtime {
                        event: *event,
                        target: Box::new(target.clone()),
                        error: Box::new(error),
                    });
                }
            };
            if matches!(subject, EventSubject::MapAtom(_))
                && event_kind(*event) == LifecycleKind::New
                && state
                    .heap()
                    .datum_field(
                        datum,
                        &FieldName::parse("flags_1")
                            .expect("project atom initialization flag field is valid"),
                    )
                    .ok()
                    .and_then(Value::as_number)
                    .is_some_and(|flags| (flags as i32) & (1 << 7) != 0)
            {
                initialized_during_new.insert(datum);
            }
            result.executed_events += 1;
            *result
                .executed_event_counts
                .entry(event_kind(*event))
                .or_default() += 1;
            if !release_runtime_metadata {
                result.events.push(ExecutedLifecycleEvent {
                    event: *event,
                    datum,
                    procedure_path: target.procedure_path.clone(),
                    result: value,
                });
            }
        }
        eprintln!("boot-progress: completed lifecycle events");
        if !audit_failures.is_empty() {
            eprintln!(
                "boot-audit-summary: unique_runtime_failure_groups={} failed_datums={}",
                audit_failures.len(),
                failed_datums.len(),
            );
            return Err(InitializationExecutionError::AuditFailures {
                failures: audit_failures.len(),
            });
        }
        if release_runtime_metadata {
            let released = state.release_host_value_roots();
            eprintln!("boot-progress: released consumed host result roots={released}");
        }
        result.scheduler = drain_startup_scheduler(
            executable,
            &mut state,
            scheduler_limits,
            readiness,
            startup_service,
        )?;
        Ok(result)
    })();
    if execution.is_ok()
        && let Some(persistent_state) = persistent_state
    {
        *persistent_state = Some(state);
    } else {
        runtime.restore_execution_state(state);
    }
    execution
}

fn drain_startup_scheduler(
    executable: &dm_semantics::ExecutableProcedures,
    state: &mut dm_vm::ExecutionState,
    limits: SchedulerDrainLimits,
    readiness: Option<&HeadlessReadinessProbe>,
    mut startup_service: Option<
        &mut dyn FnMut(&dm_semantics::ExecutableProcedures, &mut dm_vm::ExecutionState),
    >,
) -> Result<SchedulerDrain, InitializationExecutionError> {
    let start_tick = state.scheduler_tick();
    let tick_limit = start_tick.saturating_add(limits.max_ticks);
    let mut drain = SchedulerDrain {
        final_tick: start_tick,
        ..SchedulerDrain::default()
    };
    loop {
        if let Some(service) = startup_service.as_deref_mut() {
            service(executable, state);
        }
        if state.scheduled_task_count() == 0 {
            break;
        }
        if readiness.is_some_and(|probe| readiness_probe_matches(state, probe)) {
            drain.termination = SchedulerDrainTermination::HeadlessReady;
            break;
        }
        if drain.rounds >= limits.max_rounds {
            drain.termination = SchedulerDrainTermination::RoundLimit;
            break;
        }
        let next_tick = state
            .next_scheduled_tick()
            .expect("a non-empty scheduler has an earliest task");
        if next_tick > tick_limit {
            drain.termination = SchedulerDrainTermination::TickLimit;
            break;
        }
        let advance = next_tick.saturating_sub(state.scheduler_tick());
        match advance_scheduler(
            executable.module(),
            advance,
            ExecutionLimits::default(),
            state,
        ) {
            Ok(completed) => {
                drain.rounds += 1;
                drain.completed_tasks += completed.len();
                drop(completed);
            }
            Err(error) => {
                if startup_fail_fast_on_error() {
                    return Err(InitializationExecutionError::Scheduler(error));
                }
                drain.rounds += 1;
                drain.failed_tasks = drain.failed_tasks.saturating_add(1);
                eprintln!(
                    "startup-runtime: isolated scheduled thread failure (continuing): {error}"
                );
            }
        }
        state.release_host_value_roots();
        drain.final_tick = state.scheduler_tick();
        if drain.rounds == 1 || drain.rounds % 1000 == 0 {
            eprintln!(
                "boot-progress: startup-scheduler slice={} tick={} completed={} failed={} pending={}",
                drain.rounds,
                drain.final_tick,
                drain.completed_tasks,
                drain.failed_tasks,
                state.scheduled_task_count()
            );
        }
    }
    drain.pending_tasks = state.scheduled_task_count();
    if readiness.is_some_and(|probe| readiness_probe_matches(state, probe)) {
        drain.termination = SchedulerDrainTermination::HeadlessReady;
    } else if drain.pending_tasks == 0 {
        drain.termination = SchedulerDrainTermination::StableIdle;
    }
    eprintln!(
        "boot-progress: scheduler termination={:?} tick={} rounds={} completed={} pending={}",
        drain.termination,
        drain.final_tick,
        drain.rounds,
        drain.completed_tasks,
        drain.pending_tasks
    );
    if !matches!(
        drain.termination,
        SchedulerDrainTermination::HeadlessReady | SchedulerDrainTermination::StableIdle
    ) {
        for line in state.bounded_scheduler_progress(executable.module()) {
            eprintln!("boot-progress: bounded-dm-frame {line}");
        }
    }
    Ok(drain)
}

fn startup_fail_fast_on_error() -> bool {
    static STARTUP_CONTINUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STARTUP_CONTINUE.get_or_init(|| {
        env::var_os("DREAM64_STRICT_STARTUP_ERRORS").is_some()
            || env::var_os("DREAM64_FAIL_FAST_STARTUP_ERRORS").is_some()
            || env::var_os("DREAM64_STARTUP_FATAL").is_some()
    })
}

/// Advances persistent scheduled server work in a bounded host-loop slice.
/// Pending continuations remain in the runtime image for the next slice.
pub fn advance_persistent_scheduler(
    precompiled: &mut PrecompiledLifecycle,
    _runtime: &mut RuntimeImage,
    limits: SchedulerDrainLimits,
) -> Result<SchedulerDrain, InitializationExecutionError> {
    let mut state = precompiled
        .persistent_state
        .take()
        .expect("persistent scheduler requires completed precompiled lifecycle execution");
    let result = drain_persistent_scheduler(
        precompiled.executable.module(),
        &mut state,
        limits,
        ExecutionLimits::default(),
    );
    precompiled.persistent_state = Some(state);
    Ok(result)
}

/// Advances persistent work with an instruction-bounded cooperative dispatch.
/// Budget exhaustion retains the scheduled continuation at the same tick, so
/// the host can service transport queues before resuming exact VM state.
pub fn advance_persistent_scheduler_responsive(
    precompiled: &mut PrecompiledLifecycle,
    _runtime: &mut RuntimeImage,
    limits: SchedulerDrainLimits,
    max_steps_per_round: u64,
) -> Result<SchedulerDrain, InitializationExecutionError> {
    let mut state = precompiled
        .persistent_state
        .take()
        .expect("persistent scheduler requires completed precompiled lifecycle execution");
    let result = drain_persistent_scheduler(
        precompiled.executable.module(),
        &mut state,
        limits,
        ExecutionLimits {
            max_steps: max_steps_per_round.max(1),
            ..ExecutionLimits::default()
        },
    );
    precompiled.persistent_state = Some(state);
    Ok(result)
}

fn drain_persistent_scheduler(
    module: &dm_vm::Module,
    state: &mut dm_vm::ExecutionState,
    limits: SchedulerDrainLimits,
    execution_limits: ExecutionLimits,
) -> SchedulerDrain {
    let start_tick = state.scheduler_tick();
    let tick_limit = start_tick.saturating_add(limits.max_ticks);
    let mut drain = SchedulerDrain {
        final_tick: start_tick,
        ..SchedulerDrain::default()
    };

    while state.scheduled_task_count() != 0 {
        if drain.rounds >= limits.max_rounds {
            drain.termination = SchedulerDrainTermination::RoundLimit;
            break;
        }
        let next_tick = state
            .next_scheduled_tick()
            .expect("a non-empty scheduler has an earliest task");
        if next_tick > tick_limit {
            drain.termination = SchedulerDrainTermination::TickLimit;
            break;
        }
        let advance = next_tick.saturating_sub(state.scheduler_tick());
        drain.rounds = drain.rounds.saturating_add(1);
        match advance_scheduler(module, advance, execution_limits, state) {
            Ok(completed) => {
                drain.completed_tasks = drain.completed_tasks.saturating_add(completed.len());
                drop(completed);
                state.release_host_value_roots();
            }
            Err(error) => {
                // `advance_scheduler` drops only the failing continuation and
                // restores every later due task to scheduler state. Match the
                // server scheduler's thread isolation here: report the full
                // source-mapped failure, then keep draining the other work.
                drain.failed_tasks = drain.failed_tasks.saturating_add(1);
                state.release_host_value_roots();
                eprintln!(
                    "server-runtime: isolated scheduled thread failure (continuing): {error}"
                );
            }
        }
        drain.final_tick = state.scheduler_tick();
    }

    // A persistent server owns a clock even when no DM continuation is
    // pending. Advance an otherwise idle/between-task slice to its bounded
    // tick boundary. RoundLimit is the exception: same-tick work must retain
    // its tick and source order for the next host slice.
    if drain.termination != SchedulerDrainTermination::RoundLimit
        && state.scheduler_tick() < tick_limit
    {
        drain.rounds = drain.rounds.saturating_add(1);
        let completed = advance_scheduler(
            module,
            tick_limit.saturating_sub(state.scheduler_tick()),
            execution_limits,
            state,
        )
        .expect("no scheduled task is due before the validated persistent tick boundary");
        drain.completed_tasks = drain.completed_tasks.saturating_add(completed.len());
        drop(completed);
        state.release_host_value_roots();
        drain.final_tick = state.scheduler_tick();
    }

    drain.pending_tasks = state.scheduled_task_count();
    if drain.pending_tasks == 0 {
        drain.termination = SchedulerDrainTermination::StableIdle;
    } else if drain.termination != SchedulerDrainTermination::RoundLimit {
        drain.termination = SchedulerDrainTermination::TickLimit;
    }
    drain
}

fn readiness_probe_matches(state: &dm_vm::ExecutionState, probe: &HeadlessReadinessProbe) -> bool {
    let storage = probe.qualified_storage.as_ref().unwrap_or(&probe.global);
    let Some(mut value) = state.global(storage).cloned() else {
        return false;
    };
    for field in &probe.fields {
        let Value::Datum(datum) = value else {
            return false;
        };
        let Ok(next) = state.heap().datum_field(datum, field) else {
            return false;
        };
        value = next.clone();
    }
    value == probe.expected
}

/// Compiles every lifecycle-reachable body independently and retains all
/// VM-subset failures instead of stopping at the first one.
///
/// Unlike boot this fast inventory deliberately does not follow calls or
/// `..()` targets. Use [`sweep_lifecycle_compatibility_with_closures`] for the
/// slower boot-equivalent audit.
#[must_use]
pub fn sweep_lifecycle_compatibility(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    plan: &InitializationPlan,
) -> LifecycleCompatibilitySweep {
    sweep_lifecycle_compatibility_inner(compilation, procedures, index, plan, false)
}

/// Compiles every lifecycle target with its boot-time dependency closure and
/// retains all VM-subset failures.
///
/// This is more complete than [`sweep_lifecycle_compatibility`], but can be
/// substantially slower for large projects because closures overlap.
#[must_use]
pub fn sweep_lifecycle_compatibility_with_closures(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    plan: &InitializationPlan,
) -> LifecycleCompatibilitySweep {
    sweep_lifecycle_compatibility_inner(compilation, procedures, index, plan, true)
}

fn sweep_lifecycle_compatibility_inner(
    compilation: &Compilation,
    procedures: &ProcedureRegistry,
    index: &LifecycleIndex,
    plan: &InitializationPlan,
    with_closures: bool,
) -> LifecycleCompatibilitySweep {
    let mut locations =
        BTreeMap::<ProcedureImplementationId, Vec<LifecycleCompatibilityLocation>>::new();
    for event in &plan.events {
        let Some(target) = event_target(*event, index) else {
            continue;
        };
        let location = LifecycleCompatibilityLocation {
            kind: event_kind(*event),
            procedure_path: target.procedure_path.clone(),
            source: target.source.clone(),
        };
        let entry = locations.entry(target.implementation).or_default();
        if !entry.contains(&location) {
            entry.push(location);
        }
    }

    let targets = locations.len();
    let mut compatible = 0;
    let mut grouped = BTreeMap::<(String, String), Vec<LifecycleCompatibilityLocation>>::new();
    let results = if with_closures {
        locations
            .keys()
            .copied()
            .map(|implementation| {
                (
                    implementation,
                    procedures.compile_vm_implementations(compilation, [implementation]),
                )
            })
            .collect()
    } else {
        procedures.compile_vm_bodies_independently(compilation, locations.keys().copied())
    };
    for (implementation, result) in results {
        let target_locations = locations
            .remove(&implementation)
            .expect("sweep result should retain its selected target");
        match result {
            Ok(_) => compatible += 1,
            Err(error) => {
                let message = error.message;
                let category = compatibility_category(&message);
                grouped
                    .entry((category, message))
                    .or_default()
                    .extend(target_locations);
            }
        }
    }
    let issues = grouped
        .into_iter()
        .map(|((category, message), mut locations)| {
            locations.sort_by(|left, right| {
                (
                    left.source.path.as_str(),
                    left.source.span.start,
                    left.procedure_path.as_str(),
                    left.kind,
                )
                    .cmp(&(
                        right.source.path.as_str(),
                        right.source.span.start,
                        right.procedure_path.as_str(),
                        right.kind,
                    ))
            });
            LifecycleCompatibilityIssue {
                category,
                message,
                locations,
            }
        })
        .collect();
    LifecycleCompatibilitySweep {
        targets,
        compatible,
        issues,
    }
}

fn compatibility_category(message: &str) -> String {
    message
        .split_once(':')
        .map_or_else(|| message.to_owned(), |(category, _)| category.to_owned())
}

fn apply_dynamic_map_overrides(
    plan: &InitializationPlan,
    allocation: &WorldAllocation,
    bindings: &[Option<DatumId>],
    runtime: &RuntimeImage,
    state: &mut dm_vm::ExecutionState,
    module: &mut dm_vm::Module,
) -> Result<(), InitializationExecutionError> {
    for (atom_index, atom) in plan.map_atoms.iter().enumerate() {
        let Some(datum) = bindings.get(atom_index).and_then(|datum| *datum) else {
            continue;
        };
        for assignment in atom.variables.iter() {
            if !is_dynamic_map_assignment(allocation, atom, assignment) {
                continue;
            }
            let field = FieldName::parse(&assignment.name).map_err(|error| {
                InitializationExecutionError::MapExpression {
                    atom_index,
                    field: assignment.name.clone(),
                    span: assignment.span,
                    error: Box::new(dm_runtime::RuntimeImageError::Value(error)),
                }
            })?;
            let value = runtime
                .evaluate_datum_expression_linked(datum, &assignment.value.raw, state, module)
                .map_err(|error| InitializationExecutionError::MapExpression {
                    atom_index,
                    field: assignment.name.clone(),
                    span: assignment.span,
                    error: Box::new(error),
                })?;
            state
                .heap_mut()
                .set_datum_field(datum, field, value)
                .map_err(|error| InitializationExecutionError::MapExpression {
                    atom_index,
                    field: assignment.name.clone(),
                    span: assignment.span,
                    error: Box::new(dm_runtime::RuntimeImageError::Value(error)),
                })?;
        }
    }
    Ok(())
}

fn is_dynamic_map_assignment(
    allocation: &WorldAllocation,
    atom: &PlannedAtom,
    assignment: &MapVariableAssignment,
) -> bool {
    allocation.work_items().iter().any(|item| {
        matches!(item.kind, WorldAllocationWorkKind::DynamicOverride(_))
            && item.coordinate == atom.placement.coordinate
            && item.initializer_path.as_deref() == Some(atom.type_path.as_ref())
            && item.field.as_deref() == Some(assignment.name.as_str())
            && item.raw_value.as_deref() == Some(assignment.value.raw.as_str())
    })
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
                    .is_ok_and(|record| record.type_path().as_str() == atom.type_path.as_ref())
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

struct SharedMapInitializer {
    type_path: Arc<str>,
    variables: Arc<[MapVariableAssignment]>,
}

struct SharedMapTemplate {
    key: Arc<str>,
    initializers: Vec<SharedMapInitializer>,
}

fn plan_map_atoms(
    index: &LifecycleIndex,
    world: &WorldPlan,
    map_path: &str,
    diagnostics: &mut Vec<LifecycleDiagnostic>,
) -> Vec<PlannedAtom> {
    let shared_map_path = Arc::<str>::from(map_path);
    let mut shared_type_paths = BTreeMap::<&str, Arc<str>>::new();
    let shared_templates = world
        .templates()
        .iter()
        .map(|(key, template)| {
            let initializers = template
                .initializers
                .iter()
                .map(|initializer| {
                    let type_path = Arc::clone(
                        shared_type_paths
                            .entry(initializer.path.as_str())
                            .or_insert_with(|| Arc::from(initializer.path.as_str())),
                    );
                    SharedMapInitializer {
                        type_path,
                        variables: Arc::from(initializer.variables.clone()),
                    }
                })
                .collect();
            (
                key.as_str(),
                SharedMapTemplate {
                    key: Arc::from(key.as_str()),
                    initializers,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
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
        let shared_template = shared_templates
            .get(cell.key.as_str())
            .expect("retained world templates have shared lifecycle metadata");
        for (initializer_index, initializer) in template.initializers.iter().enumerate() {
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
            let shared_initializer = &shared_template.initializers[initializer_index];
            map_atoms.push(PlannedAtom {
                type_index,
                type_path: Arc::clone(&shared_initializer.type_path),
                category,
                variables: Arc::clone(&shared_initializer.variables),
                placement: MapPlacementContext {
                    map_path: Arc::clone(&shared_map_path),
                    key: Arc::clone(&shared_template.key),
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
            return LifecycleResolution::Unsupported(Arc::new(LifecycleTargetIssue {
                kind: LifecycleTargetIssueKind::MissingEffectiveTarget,
                message: "type inheritance cycle prevents lifecycle resolution".to_owned(),
                procedure_path: None,
            }));
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
        return LifecycleResolution::Unsupported(Arc::new(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingEffectiveTarget,
            message: format!("{procedure_path} has no effective implementation"),
            procedure_path: Some(procedure_path),
        }));
    };
    let Some(target) = registry.implementation(target_id) else {
        return LifecycleResolution::Unsupported(Arc::new(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingImplementation,
            message: format!("effective implementation for {procedure_path} is absent"),
            procedure_path: Some(procedure_path),
        }));
    };
    if compilation
        .syntax(target.file_id)
        .and_then(|syntax| syntax.definitions.get(target.definition_index))
        .is_none()
    {
        return LifecycleResolution::Unsupported(Arc::new(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingSourceDefinition,
            message: format!("source definition for {procedure_path} is absent"),
            procedure_path: Some(procedure_path),
        }));
    }
    let Some(file) = compilation.project().file(target.file_id) else {
        return LifecycleResolution::Unsupported(Arc::new(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingSourceDefinition,
            message: format!("source file for {procedure_path} is absent"),
            procedure_path: Some(procedure_path),
        }));
    };
    let declaring_node = procedure.owner_type.unwrap_or(requested_type);
    let declaring_type = compilation
        .code_tree()
        .node(declaring_node)
        .map_or_else(|| "<unknown>".to_owned(), |node| node.path.to_string());
    LifecycleResolution::Resolved(Arc::new(LifecycleTarget {
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
    }))
}

fn share_resolved_lifecycle_targets(types: &mut [TypeLifecycle]) {
    let mut shared = BTreeMap::<(ProcedureImplementationId, bool), Arc<LifecycleTarget>>::new();
    for lifecycle in types {
        for resolution in [
            &mut lifecycle.targets.genesis,
            &mut lifecycle.targets.new_target,
            &mut lifecycle.targets.initialize,
            &mut lifecycle.targets.late_initialize,
            &mut lifecycle.targets.destroy,
        ] {
            let LifecycleResolution::Resolved(target) = resolution else {
                continue;
            };
            let key = (target.implementation, target.inherited);
            if let Some(existing) = shared.get(&key) {
                *target = Arc::clone(existing);
            } else {
                shared.insert(key, Arc::clone(target));
            }
        }
    }
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
        && has_target(index, type_index, LifecycleKind::Genesis)
    {
        events.push(InitializationEvent::Lifecycle {
            subject: EventSubject::World,
            kind: LifecycleKind::Genesis,
            type_index,
        });
    }
    // BYOND constructs every compiled-map atom after global/Genesis bootstrap
    // but before world/New. Tg/Monk's SSatoms then owns Initialize and
    // LateInitialize during Master subsystem startup. Running those hooks
    // synthetically before the Master task is drained observes uninitialized
    // subsystems (greyscale, materials, lighting, ...), producing cascades that
    // cannot occur in the real engine pipeline.
    for (atom_index, atom) in atoms.iter().enumerate() {
        if has_target(index, atom.type_index, LifecycleKind::New) {
            events.push(InitializationEvent::Lifecycle {
                subject: EventSubject::MapAtom(atom_index),
                kind: LifecycleKind::New,
                type_index: atom.type_index,
            });
        }
    }
    if let Some(type_index) = world_type
        && has_target(index, type_index, LifecycleKind::New)
    {
        events.push(InitializationEvent::Lifecycle {
            subject: EventSubject::World,
            kind: LifecycleKind::New,
            type_index,
        });
    }
    if !project_manages_atom_initialization(index) {
        for kind in [LifecycleKind::Initialize, LifecycleKind::LateInitialize] {
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
    }
    events
}

fn project_manages_atom_initialization(index: &LifecycleIndex) -> bool {
    index
        .find_path("/datum/controller/subsystem/atoms")
        .is_some_and(|lifecycle| {
            matches!(
                lifecycle.targets.get(LifecycleKind::Initialize),
                LifecycleResolution::Resolved(_)
            )
        })
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use dm_compiler::{Compilation, CompilerDatabase};
    use dm_map::parse;
    use dm_runtime::RuntimeImage;
    use dm_semantics::{ExecutableProcedures, ProcedureRegistry};
    use dm_value::{FieldName, TypePath, Value};
    use dm_vm::{
        ExecutionContext, ExecutionState, execute_module_in_context, execute_module_in_state,
    };
    use dm_world::{WorldCoordinate, allocate_world, build_plan};

    use super::{
        EventSubject, HeadlessReadinessProbe, HostSliceBudget, InitializationEvent,
        InitializationExecutionError, LifecycleIndex, LifecycleKind, LifecycleResolution,
        SchedulerDrainLimits, SchedulerDrainTermination, advance_persistent_scheduler,
        audit_initialization_plan_with_precompiled, build_initialization_plan, construct_datum,
        delete_datum, execute_boot_initialization_plan_with_precompiled_and_startup_service,
        execute_initialization_plan, execute_initialization_plan_with_precompiled,
        precompile_lifecycle_for_world, sweep_lifecycle_compatibility,
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

    #[test]
    fn cached_lobby_lifecycle_index_does_not_force_procedure_dependencies() {
        let (_fixture, compilation) = Fixture::compile(concat!(
            "/world/New()\n\treturn ..()\n",
            "/client/New()\n\treturn ..()\n",
            "/datum/example/proc/run()\n\treturn 1\n",
        ));
        let procedures = ProcedureRegistry::build_lazy(&compilation);
        assert!(!procedures.dependencies_initialized());
        let index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        assert!(index.find_path("/world").is_some());
        assert!(!procedures.dependencies_initialized());
    }

    #[test]
    fn artifact_backed_precompile_does_not_rebuild_procedure_dependencies() {
        let (_fixture, compilation) = Fixture::compile(concat!(
            "/world/New()\n\treturn ..()\n",
            "/area/test\n/turf/test\n/obj/test\n\tNew()\n\t\treturn ..()\n",
        ));
        let eager = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let eager_index = LifecycleIndex::build_compile_only(&compilation, &eager);
        let roots = crate::lifecycle_targets_for_world(&eager_index, &world);
        let executable = eager
            .compile_vm_all_symbolic_with_eager_roots(&compilation, roots.iter().copied())
            .expect("fixture executable should link");

        let lazy = ProcedureRegistry::build_lazy(&compilation);
        let lazy_index = LifecycleIndex::build_compile_only(&compilation, &lazy);
        assert!(!lazy.dependencies_initialized());
        let precompiled = crate::precompile_lifecycle_for_world_with_executable(
            &compilation,
            &lazy,
            &lazy_index,
            &world,
            executable,
        );
        assert!(!lazy.dependencies_initialized());
        assert_eq!(
            precompiled.reachable_bodies(),
            precompiled.module_procedures()
        );
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
        assert!(
            std::mem::size_of::<LifecycleResolution>() <= 2 * std::mem::size_of::<usize>(),
            "every type retains five lifecycle resolution slots, so payloads must remain indirect"
        );
        let (_fixture, _compilation, _runtime, index) = index(
            "/datum/base\n\tproc/New()\n\tproc/Initialize()\n\tproc/Destroy()\n/datum/base/child\n\tInitialize()\n\tproc/LateInitialize()\n/datum/base/sibling\n",
        );
        let child = index
            .find_path("/datum/base/child")
            .expect("child lifecycle should exist");
        let sibling = index
            .find_path("/datum/base/sibling")
            .expect("sibling lifecycle should exist");

        let LifecycleResolution::Resolved(new_target) = &child.targets.new_target else {
            panic!("New should resolve");
        };
        assert!(new_target.inherited);
        assert_eq!(new_target.declaring_type, "/datum/base");
        let LifecycleResolution::Resolved(sibling_new) = &sibling.targets.new_target else {
            panic!("sibling New should resolve");
        };
        assert!(Arc::ptr_eq(new_target, sibling_new));
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
    fn runtime_upcast_dereference_uses_derived_initial_fields() {
        let source = "/datum/later\n\tvar/datum/pointless_base/a\n/datum/pointless_base/derived/var/x = 7\n/proc/RunTest()\n\tvar/datum/later/L = new\n\tL.a = new /datum/pointless_base/derived()\n\treturn L.a:x\n";
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let target = procedures
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == "/proc/RunTest")
            .and_then(|procedure| procedure.effective_target)
            .expect("RunTest should have an effective implementation");
        let executable = procedures
            .compile_vm_implementations(&compilation, [target])
            .expect("RunTest should lower");
        let entry = executable
            .implementation(target)
            .expect("RunTest should be in the VM module");
        let mut runtime = RuntimeImage::from_compilation(&compilation)
            .expect("runtime image should materialize defaults");
        let mut state = runtime.take_execution_state();
        assert_eq!(
            execute_module_in_state(executable.module(), entry, &[], &mut state),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn construction_orders_defaults_parent_new_and_arguments_and_cleans_failures() {
        let source = "/datum/base\n\tvar/value = 1\n\tvar/stage = 1\n\tvar/seen_default = 0\n\tvar/list/waiting_calls\n\tNew(arg)\n\t\tseen_default = value\n\t\tstage = stage * 10 + arg\n/datum/base/sub\n\tvalue = 7\n\tNew(arg)\n\t\t..()\n\t\tstage = stage * 10 + 2\n/datum/fail/New()\n\tvar/list/L = null\n\treturn L[1]\n";
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let mut runtime = RuntimeImage::from_compilation(&compilation)
            .expect("runtime image should materialize defaults");
        let subtype = TypePath::parse("/datum/base/sub").unwrap();
        let datum = construct_datum(
            &compilation,
            &procedures,
            &mut runtime,
            &subtype,
            &[Value::number(3.0)],
        )
        .expect("subtype constructor should run");
        let record = runtime.heap().datum(datum).unwrap();
        assert_eq!(
            record.field(&FieldName::parse("seen_default").unwrap()),
            Ok(&Value::number(7.0))
        );
        assert_eq!(
            record.field(&FieldName::parse("stage").unwrap()),
            Ok(&Value::number(132.0))
        );
        assert_eq!(
            record.field(&FieldName::parse("waiting_calls").unwrap()),
            Ok(&Value::Null),
            "plain inherited declarations must exist before New runs"
        );

        let before = runtime.heap().datums().count();
        let failure = TypePath::parse("/datum/fail").unwrap();
        assert!(construct_datum(&compilation, &procedures, &mut runtime, &failure, &[]).is_err());
        assert_eq!(
            runtime.heap().datums().count(),
            before,
            "failed constructor allocation must be destroyed"
        );
    }

    #[test]
    fn deletion_runs_parent_cleanup_once_and_invalidates_on_failure() {
        let source = "var/global/events = 0\n/datum/base/Destroy()\n\tevents = events * 10 + 1\n/datum/base/sub/Destroy()\n\t..()\n\tevents = events * 10 + 2\n/datum/fail/Destroy()\n\tvar/list/L = null\n\treturn L[1]\n/datum/reentrant/Destroy()\n\tqdel(src)\n";
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let mut runtime =
            RuntimeImage::from_compilation(&compilation).expect("runtime image should materialize");
        let subtype = TypePath::parse("/datum/base/sub").unwrap();
        let datum = construct_datum(&compilation, &procedures, &mut runtime, &subtype, &[])
            .expect("datum should construct");
        delete_datum(&compilation, &procedures, &mut runtime, datum)
            .expect("cleanup chain should succeed");
        assert!(
            runtime.heap().datum(datum).is_err(),
            "deleted handle must be stale"
        );
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/events"))
                .map(|variable| &variable.value),
            Some(&Value::number(12.0)),
            "parent Destroy must run before subtype cleanup exactly once"
        );

        let failure = TypePath::parse("/datum/fail").unwrap();
        let failing = construct_datum(&compilation, &procedures, &mut runtime, &failure, &[])
            .expect("failing-cleanup datum should construct");
        assert!(delete_datum(&compilation, &procedures, &mut runtime, failing).is_err());
        assert!(
            runtime.heap().datum(failing).is_err(),
            "cleanup failure must still invalidate the datum"
        );

        let reentrant = TypePath::parse("/datum/reentrant").unwrap();
        let reentrant_datum =
            construct_datum(&compilation, &procedures, &mut runtime, &reentrant, &[])
                .expect("reentrant-cleanup datum should construct");
        delete_datum(&compilation, &procedures, &mut runtime, reentrant_datum)
            .expect("qdel(src) during cleanup should count as already deleted");
        assert!(runtime.heap().datum(reentrant_datum).is_err());
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
            "\"a\" = (/obj/test{name = \"crate\"; dir = 4}, /turf/test, /area/test)\n",
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
        assert_eq!(plan.map_atoms[0].placement.map_path.as_ref(), "test.dmm");
        assert_eq!(plan.map_atoms[0].variables.len(), 2);
        assert_eq!(plan.map_atoms[0].variables[0].name, "name");
        assert_eq!(plan.map_atoms[0].variables[0].value.raw, "\"crate\"");
        assert_eq!(plan.map_atoms[0].variables[1].name, "dir");
        assert_eq!(plan.map_atoms[0].variables[1].raw, "dir = 4");
        assert!(
            plan.map_atoms[0].variables[0].name_span.start
                < plan.map_atoms[0].variables[0].span.end
        );
        assert_eq!(plan.events[0], InitializationEvent::Globals);
        assert!(matches!(
            plan.events[1],
            InitializationEvent::Lifecycle {
                subject: EventSubject::MapAtom(_),
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
    fn repeated_map_template_rows_share_immutable_planning_metadata() {
        let source = concat!(
            "/world/New()\n",
            "/area/test\n",
            "/turf/test\n",
            "/obj/test\n\tInitialize()\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test{name = \"crate\"; dir = 4}, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\naaa\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "shared-map-path.dmm");

        assert_eq!(plan.map_atoms.len(), 9);
        let first = &plan.map_atoms[0];
        let second = &plan.map_atoms[3];
        let third = &plan.map_atoms[6];
        assert_ne!(first.placement.coordinate, second.placement.coordinate);
        assert_ne!(second.placement.coordinate, third.placement.coordinate);
        assert!(Arc::ptr_eq(
            &first.placement.map_path,
            &second.placement.map_path,
        ));
        assert!(Arc::ptr_eq(
            &second.placement.map_path,
            &third.placement.map_path,
        ));
        assert!(Arc::ptr_eq(&first.placement.key, &second.placement.key));
        assert!(Arc::ptr_eq(&second.placement.key, &third.placement.key));
        assert!(Arc::ptr_eq(&first.type_path, &second.type_path));
        assert!(Arc::ptr_eq(&second.type_path, &third.type_path));
        assert!(Arc::ptr_eq(&first.variables, &second.variables));
        assert!(Arc::ptr_eq(&second.variables, &third.variables));
        assert_eq!(first.variables.len(), 2);
        assert_eq!(first.variables[0].span, second.variables[0].span);
        assert_eq!(
            second.variables[1].value.span,
            third.variables[1].value.span
        );
        assert_eq!(first, &plan.clone().map_atoms[0]);
    }

    #[test]
    fn monk_pipeline_constructs_compiled_atoms_before_world_new_and_defers_init_to_ssatoms() {
        let source = concat!(
            "/world/Genesis()\n/world/New()\n",
            "/atom/New(loc)\n/atom/Initialize()\n/atom/LateInitialize()\n",
            "/datum/controller/subsystem/atoms/Initialize()\n",
            "/area/test\n/turf/test\n/obj/test\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("one-cell subsystem-managed map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "managed.dmm");
        let lifecycle: Vec<_> = plan
            .events
            .iter()
            .filter_map(|event| match event {
                InitializationEvent::Lifecycle { subject, kind, .. } => Some((*subject, *kind)),
                InitializationEvent::Globals => None,
            })
            .collect();

        assert_eq!(lifecycle[0], (EventSubject::World, LifecycleKind::Genesis));
        let world_new = lifecycle
            .iter()
            .position(|event| *event == (EventSubject::World, LifecycleKind::New))
            .expect("world New should be planned");
        assert!(
            lifecycle[1..world_new]
                .iter()
                .all(
                    |(subject, kind)| matches!(subject, EventSubject::MapAtom(_))
                        && *kind == LifecycleKind::New
                )
        );
        assert!(lifecycle[world_new + 1..].is_empty());
        assert!(!lifecycle.iter().any(|(_, kind)| matches!(
            kind,
            LifecycleKind::Initialize | LifecycleKind::LateInitialize
        )));
    }

    #[test]
    fn executes_map_lifecycles_in_phase_order_without_compiling_unrelated_procs() {
        let source = concat!(
            "var/global/lifecycle_count = 1\n",
            "/world/New()\n\tsrc.stage = 5\n\tglobal.lifecycle_count += 1\n",
            "/atom/proc/New(loc)\n\tsrc.stage = (args.len * 10) + (args[1] == src.loc)\n",
            "/atom/proc/Initialize()\n\tsrc.stage += 1\n\tglobal.lifecycle_count += 1\n",
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
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/lifecycle_count"))
                .expect("global should remain materialized")
                .value,
            Value::number(5.0)
        );
        for datum in allocation.allocation_order() {
            assert_eq!(
                runtime.heap().datum_field(*datum, &stage),
                Ok(&Value::number(112.0))
            );
        }
    }

    #[test]
    fn map_new_that_initializes_through_ssatoms_is_not_initialized_twice() {
        let source = concat!(
            "var/global/initialize_count = 0\n",
            "/world/New()\n",
            "/atom\n\tvar/flags_1 = 0\n\tvar/stage = 0\n",
            "/atom/proc/New(loc)\n\tsrc.Initialize(1)\n",
            "/atom/proc/Initialize(mapload)\n\tif(flags_1 & 128)\n\t\treturn -1\n\tflags_1 |= 128\n\tstage += 1\n\tglobal.initialize_count += 1\n\treturn 0\n",
            "/atom/proc/LateInitialize()\n\tstage += 100\n",
            "/area/test\n/turf/test\n/obj/test\n",
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
        .expect("immediate initialization should not be repeated");

        assert_eq!(execution.events.len(), 4, "world New plus three atom News");
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/initialize_count"))
                .expect("counter global")
                .value,
            Value::number(3.0),
        );
        for datum in allocation.allocation_order() {
            assert_eq!(
                runtime
                    .heap()
                    .datum_field(*datum, &FieldName::parse("stage").expect("stage field"),),
                Ok(&Value::number(1.0)),
                "synthetic Initialize/LateInitialize must be skipped after New initialized it",
            );
        }
    }

    #[test]
    fn precompiled_lifecycle_links_dynamic_map_expressions_to_project_procs() {
        let source = concat!(
            "/proc/map_value()\n\treturn 37\n",
            "/area/test\n/turf/test\n/obj/test\n\tvar/value = 0\n\tNew()\n\t\tmap_value()\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test{value = map_value()}, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("lifecycle should precompile without a runtime image");

        let mut runtime =
            RuntimeImage::from_compilation(&compilation).expect("runtime image should materialize");
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
        execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            None,
            &mut precompiled,
        )
        .expect("precompiled lifecycle and linked map expression should execute");

        let object = allocation
            .allocation_order()
            .iter()
            .copied()
            .find(|datum| {
                runtime
                    .heap()
                    .datum(*datum)
                    .is_ok_and(|record| record.type_path().as_str() == "/obj/test")
            })
            .expect("mapped object should exist");
        assert_eq!(
            runtime
                .heap()
                .datum_field(object, &FieldName::parse("value").unwrap()),
            Ok(&Value::number(37.0))
        );
    }

    #[test]
    fn precompiled_global_initializer_family_smoke_executes_transitive_file_constructor() {
        let source = concat!(
            "var/global/datum/controller/global_vars/GLOB\n",
            "var/global/smoke = \"\"\n",
            "/proc/trim(value)\n\treturn trimtext(value)\n",
            "/proc/file2list(filename, separator = \"\\n\", trim_file = TRUE)\n",
            "\tif(trim_file)\n",
            "\t\treturn splittext(trim(file2text(filename)), separator)\n",
            "\treturn splittext(file2text(filename), separator)\n",
            "/datum/controller/global_vars\n",
            "\tvar/datum/advertisements/advertisements\n",
            "\tproc/InitGlobaladvertisements()\n",
            "\t\tadvertisements = new\n",
            "\tproc/Initialize()\n",
            "\t\tfor(var/global_init in typesof(/datum/controller/global_vars/proc))\n",
            "\t\t\tif(global_init == /datum/controller/global_vars/proc/Initialize)\n",
            "\t\t\t\tcontinue\n",
            "\t\t\tcall(src, global_init)()\n",
            "/datum/advertisements\n",
            "\tvar/result = \"\"\n",
            "\tNew()\n",
            "\t\tresult = load_file(\"advertisements.txt\")\n",
            "\tproc/load_file(filename)\n",
            "\t\tvar/list/lines = file2list(filename)\n",
            "\t\tvar/output = \"\"\n",
            "\t\tfor(var/line in lines)\n",
            "\t\t\toutput += line[1]\n",
            "\t\treturn output\n",
            "/world/Genesis()\n",
            "\tGLOB = new\n",
            "\tGLOB.Initialize()\n",
            "\tglobal.smoke = GLOB.advertisements.result\n",
            "/area/test\n/turf/test\n",
        );
        let (fixture, compilation) = Fixture::compile(source);
        fs::write(
            fixture.0.join("advertisements.txt"),
            "- advertisement separator\n",
        )
        .expect("advertisement fixture should be written");
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("smoke selector map should parse");
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let genesis = compile_index
            .find_path("/world")
            .and_then(
                |lifecycle| match lifecycle.targets.get(LifecycleKind::Genesis) {
                    LifecycleResolution::Resolved(target) => Some(target.implementation),
                    _ => None,
                },
            )
            .expect("world Genesis should resolve");
        let precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("global initializer family should precompile");
        let entry = precompiled
            .executable
            .implementation(genesis)
            .expect("Genesis should be retained by precompile");
        let mut runtime = RuntimeImage::from_compilation(&compilation)
            .expect("tiny smoke runtime should materialize");
        let world_datum = runtime
            .canonical_world()
            .expect("canonical world should exist");
        let mut state = runtime.take_execution_state();
        state.set_global(
            FieldName::parse("world").expect("world field name"),
            Value::Datum(world_datum),
        );
        execute_module_in_context(
            precompiled.executable.module(),
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(world_datum), Value::Null),
        )
        .expect("transitive generated global initializer should execute before map allocation");
        assert_eq!(
            state.global(&FieldName::parse("smoke").unwrap()),
            Some(&Value::text("-")),
        );
    }

    #[test]
    fn generated_global_qdel_executes_full_item_destroy_chain_before_map_allocation() {
        let source = concat!(
            "var/global/datum/controller/global_vars/GLOB\n",
            "var/global/destroy_trace = \"\"\n",
            "/proc/qdel(datum/to_delete)\n\treturn to_delete.Destroy()\n",
            "/datum/controller/global_vars\n",
            "\tproc/InitGlobalcleanup()\n",
            "\t\tvar/obj/item/temporary = new\n",
            "\t\tqdel(temporary)\n",
            "\tproc/Initialize()\n",
            "\t\tfor(var/global_init in typesof(/datum/controller/global_vars/proc))\n",
            "\t\t\tif(global_init == /datum/controller/global_vars/proc/Initialize)\n",
            "\t\t\t\tcontinue\n",
            "\t\t\tcall(src, global_init)()\n",
            "/datum/Destroy()\n\tglobal.destroy_trace += \"D\"\n\treturn 1\n",
            "/atom/Destroy()\n\tglobal.destroy_trace += \"A\"\n\treturn ..()\n",
            "/atom/movable/Destroy()\n\tglobal.destroy_trace += \"M\"\n\treturn ..()\n",
            "/obj/Destroy()\n",
            "\tvis_locs = null\n",
            "\tglobal.destroy_trace += \"O\"\n",
            "\treturn ..()\n",
            "/obj/item/Destroy()\n\tglobal.destroy_trace += \"I\"\n\treturn ..()\n",
            "/world/Genesis()\n",
            "\tGLOB = new /datum/controller/global_vars\n",
            "\tGLOB.Initialize()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let genesis = compile_index
            .find_path("/world")
            .and_then(
                |lifecycle| match lifecycle.targets.get(LifecycleKind::Genesis) {
                    LifecycleResolution::Resolved(target) => Some(target.implementation),
                    _ => None,
                },
            )
            .unwrap();
        let precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("generated qdel/Destroy family should precompile");
        let entry = precompiled.executable.implementation(genesis).unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let world_datum = runtime.canonical_world().unwrap();
        let mut state = runtime.take_execution_state();
        state.set_global(
            FieldName::parse("world").unwrap(),
            Value::Datum(world_datum),
        );
        execute_module_in_context(
            precompiled.executable.module(),
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(world_datum), Value::Null),
        )
        .expect("generated initializer qdel should execute every inherited Destroy body");
        assert_eq!(
            state.global(&FieldName::parse("destroy_trace").unwrap()),
            Some(&Value::text("IOMAD")),
        );
    }

    #[test]
    fn lifecycle_drains_waitfor_false_world_continuations() {
        let source = concat!(
            "var/global/finished = 0\n",
            "/world/New()\n\tset waitfor = FALSE\n\t. = 7\n\tsleep(1)\n\tglobal.finished = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
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
        .expect("detached world continuation should drain");

        assert_eq!(execution.scheduler.pending_tasks, 0);
        assert!(execution.scheduler.rounds >= 1);
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/finished"))
                .map(|variable| &variable.value),
            Some(&Value::number(1.0))
        );
    }

    #[test]
    fn readiness_and_persistent_slices_preserve_delayed_server_work() {
        let source = concat!(
            "var/global/ready = 0\nvar/global/pulses = 0\n",
            "/world/New()\n\tset waitfor = FALSE\n\tsleep(2)\n\tglobal.ready = 1\n\tsleep(20)\n\tglobal.pulses = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 10,
                max_rounds: 10,
            },
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady
        );
        assert_eq!(execution.scheduler.pending_tasks, 1);
        assert_eq!(
            precompiled.persistent_tick_duration(),
            std::time::Duration::from_millis(100)
        );

        let first = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 5,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(
            first.final_tick, 7,
            "idle slices must advance toward future work"
        );
        assert_eq!(first.pending_tasks, 1);
        for _ in 0..3 {
            advance_persistent_scheduler(
                &mut precompiled,
                &mut runtime,
                SchedulerDrainLimits {
                    max_ticks: 5,
                    max_rounds: 10,
                },
            )
            .unwrap();
        }
        let final_slice = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(final_slice.pending_tasks, 0);
        assert_eq!(
            final_slice.termination,
            SchedulerDrainTermination::StableIdle
        );
    }

    #[test]
    fn startup_service_attaches_client_before_readiness_and_preserves_session() {
        let source = concat!(
            "var/global/ready = 0\nvar/global/client_started = 0\n",
            "/world/New()\n\tset waitfor = FALSE\n\tsleep(2)\n\tglobal.ready = 1\n",
            "/client/New()\n\tglobal.client_started = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let mut attached = None;
        let mut service = |executable: &ExecutableProcedures, state: &mut ExecutionState| {
            if attached.is_none() {
                attached = Some(state.connect_local_guest(executable.module()).unwrap());
            }
        };
        let execution = execute_boot_initialization_plan_with_precompiled_and_startup_service(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 10,
                max_rounds: 20,
            },
            Some(&readiness),
            &mut precompiled,
            &mut service,
        )
        .unwrap();
        drop(service);

        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady
        );
        assert!(execution.executed_events > 0);
        assert_eq!(
            execution.executed_event_counts.values().sum::<usize>(),
            execution.executed_events,
        );
        assert!(
            execution.events.is_empty(),
            "production boot retains aggregate lifecycle counts, not per-event audit records",
        );
        let attached = attached.expect("startup service attached a client");
        let state = precompiled
            .persistent_state
            .as_ref()
            .expect("ready boot preserves persistent state");
        assert_eq!(
            state.global(&FieldName::parse("client_started").unwrap()),
            Some(&Value::number(1.0))
        );
        assert!(state.local_client_state(attached.client).is_ok());
    }

    #[test]
    fn persistent_idle_slices_keep_the_server_clock_advancing() {
        let source = concat!(
            "var/global/ready = 0\n",
            "/world/New()\n\tglobal.ready = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady
        );
        assert_eq!(execution.scheduler.pending_tasks, 0);

        let first = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(first.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(first.final_tick, 1);
        assert_eq!(first.pending_tasks, 0);

        let second = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 3,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(second.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(second.final_tick, 4);
        let state = precompiled.persistent_state.as_ref().unwrap();
        let Value::Datum(world) = state.global(&FieldName::parse("world").unwrap()).unwrap() else {
            panic!("persistent state should retain the world singleton");
        };
        assert_eq!(
            state
                .heap()
                .datum_field(*world, &FieldName::parse("time").unwrap()),
            Ok(&Value::number(4.0)),
        );
    }

    #[test]
    fn infinite_native_walk_does_not_block_readiness_or_persistent_idle_slices() {
        let source = concat!(
            "var/global/ready = 0\n",
            "var/global/walker\n",
            "/world/New()\n",
            "\tglobal.walker = new /obj/walker\n",
            "\twalk(global.walker, EAST, 1)\n",
            "\tglobal.ready = 1\n",
            "/obj/walker\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .expect("a perpetual engine walk must not prevent startup readiness");
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady,
        );
        assert_eq!(execution.scheduler.pending_tasks, 0);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .unwrap()
                .next_scheduled_tick(),
            Some(1),
        );

        for expected_tick in 1..=3 {
            let slice = advance_persistent_scheduler(
                &mut precompiled,
                &mut runtime,
                SchedulerDrainLimits {
                    max_ticks: 1,
                    max_rounds: 10,
                },
            )
            .expect("persistent walk ticks should remain bounded and non-blocking");
            assert_eq!(slice.termination, SchedulerDrainTermination::StableIdle);
            assert_eq!(slice.pending_tasks, 0);
            assert_eq!(slice.final_tick, expected_tick);
            assert_eq!(
                precompiled
                    .persistent_state
                    .as_ref()
                    .unwrap()
                    .next_scheduled_tick(),
                Some(expected_tick + 1),
            );
        }
    }

    #[test]
    fn persistent_scheduler_isolates_a_failed_thread_and_runs_later_due_work() {
        let source = concat!(
            "var/global/ready = 0\n",
            "var/global/trace = \"\"\n",
            "/proc/fail_later()\n\tCRASH(\"isolated\")\n",
            "/proc/finish_later()\n\tglobal.trace += \"L\"\n",
            "/world/New()\n",
            "\tglobal.ready = 1\n",
            "\tspawn(1) fail_later()\n",
            "\tspawn(1) finish_later()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady
        );
        assert_eq!(execution.scheduler.pending_tasks, 2);

        let slice = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 10,
            },
        )
        .expect("one failed scheduled thread must not stop the server");
        assert_eq!(slice.failed_tasks, 1);
        assert_eq!(slice.completed_tasks, 1);
        assert_eq!(slice.pending_tasks, 0);
        assert_eq!(slice.final_tick, 1);
        assert_eq!(slice.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .and_then(|state| state.global(&FieldName::parse("trace").unwrap())),
            Some(&Value::text("L")),
        );
    }

    #[test]
    fn persistent_round_limit_preserves_same_tick_work_for_the_next_slice() {
        let source = concat!(
            "var/global/ready = 0\n",
            "var/global/runs = 0\n",
            "/proc/run_again()\n",
            "\tglobal.runs += 1\n",
            "\tif(global.runs < 3)\n",
            "\t\tspawn(0) run_again()\n",
            "/world/New()\n",
            "\tglobal.ready = 1\n",
            "\tspawn(0) run_again()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();

        let bounded = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 2,
            },
        )
        .unwrap();
        assert_eq!(bounded.termination, SchedulerDrainTermination::RoundLimit);
        assert_eq!(bounded.final_tick, 0);
        assert_eq!(bounded.rounds, 2);
        assert_eq!(bounded.completed_tasks, 2);
        assert_eq!(bounded.pending_tasks, 1);

        let resumed = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 2,
            },
        )
        .unwrap();
        assert_eq!(resumed.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(resumed.final_tick, 1);
        assert_eq!(resumed.completed_tasks, 1);
        assert_eq!(resumed.pending_tasks, 0);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .and_then(|state| state.global(&FieldName::parse("runs").unwrap())),
            Some(&Value::number(3.0)),
        );
    }

    #[test]
    fn monk_like_master_stages_reach_readiness_through_waitfor_scheduler() {
        let source = concat!(
            "var/global/datum/controller/master/Master\n",
            "var/global/trace = \"\"\n",
            "var/global/ready = 0\n",
            "/proc/dispatch_initialize(target)\n",
            "\tcall(target, \"Initialize\")()\n",
            "/proc/finish_subsystem_stage()\n",
            "\tsleep(2)\n",
            "\tglobal.trace += \"S\"\n",
            "\tglobal.ready = 2\n",
            "/datum/controller/master\n",
            "\tNew()\n",
            "\t\tglobal.Master = src\n",
            "\t\tglobal.trace += \"N\"\n",
            "\tproc/Initialize()\n",
            "\t\tset waitfor = FALSE\n",
            "\t\tglobal.trace += \"I\"\n",
            "\t\tfinish_subsystem_stage()\n",
            "/world/Genesis()\n",
            "\tMaster = new /datum/controller/master\n",
            "/world/New()\n",
            "\tglobal.trace += \"W\"\n",
            "\tConfigLoaded()\n",
            "\tdispatch_initialize(Master)\n",
            "/world/proc/ConfigLoaded()\n",
            "\tglobal.trace += \"C\"\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("one-cell staged smoke map should parse");
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("Master stages should link lazily before runtime allocation");
        assert!(precompiled.deferred_procedures() > 0);

        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "staged-smoke.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(2.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 10,
                max_rounds: 20,
            },
            Some(&readiness),
            &mut precompiled,
        )
        .expect("Master staged startup should reach readiness");
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady,
        );
        assert_eq!(execution.scheduler.final_tick, 2);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .and_then(|state| state.global(&FieldName::parse("trace").unwrap())),
            Some(&Value::text("NWCIS")),
        );
    }

    #[test]
    fn genesis_infers_all_typed_global_bare_new_destinations() {
        let source = concat!(
            "var/global/datum/controller/global_vars/GLOB = null\n",
            "var/global/datum/tracy/Tracy = null\n",
            "var/global/datum/debugger/Debugger = null\n",
            "var/global/datum/log_holder/logger = null\n",
            "var/global/datum/controller/master/Master = null\n",
            "/world/Genesis()\n\tGLOB.config_error_log = \"early.log\"\n\tTracy = new\n\tDebugger = new\n\tlogger = new\n\tMaster = new\n",
            "/datum/controller/global_vars\n\tvar/global/config_error_log\n",
            "/datum/tracy\n/datum/debugger\n/datum/log_holder\n/datum/controller/master\n\tvar/static/random_seed\n",
            "/datum/controller/master/New()\n\tif(!random_seed)\n\t\trandom_seed = 29051994\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
        execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("typed global bare new should execute");

        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/config_error_log"))
                .map(|variable| &variable.value),
            Some(&Value::text("early.log")),
            "a typed global receiver must bind owner-qualified static storage even while its datum value is null; vars={:?}",
            runtime.variables(),
        );

        for (name, expected) in [
            ("Tracy", "/datum/tracy"),
            ("Debugger", "/datum/debugger"),
            ("logger", "/datum/log_holder"),
            ("Master", "/datum/controller/master"),
        ] {
            let value = &runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with(&format!("/{name}")))
                .unwrap_or_else(|| panic!("missing global {name}"))
                .value;
            let Value::Datum(datum) = value else {
                panic!("{name} should contain a datum");
            };
            assert_eq!(
                runtime.heap().datum(*datum).unwrap().type_path().as_str(),
                expected
            );
        }
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/random_seed"))
                .map(|variable| &variable.value),
            Some(&Value::number(29_051_994.0)),
        );
    }

    #[test]
    fn glob_style_datum_vars_is_live_stable_and_copies_a_snapshot() {
        let source = concat!(
            "var/global/observed = 0\n",
            "/datum/globals\n\tvar/value = 1\n\tvar/global/shared = 2\n",
            "/datum/globals/proc/TestReflection()\n\tvalue = 3\n\tvar/list/reflection = vars\n\tvar/same_proxy = (reflection == vars)\n\treflection[\"value\"] += 2\n\treflection[\"shared\"] = 7\n\tvar/list/snapshot = reflection.Copy()\n\tvalue = 9\n\tglobal.observed = same_proxy + reflection[\"value\"] + reflection[\"shared\"] + snapshot[\"value\"] + snapshot[\"shared\"]\n",
            "/world/Genesis()\n\tvar/datum/globals/controller = new\n\tcontroller.TestReflection()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse("\"a\" = (/turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n")
            .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
        execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("datum vars reflection should execute during Genesis");
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/observed"))
                .map(|variable| &variable.value),
            Some(&Value::number(29.0))
        );
    }

    #[test]
    fn sort_instance_list_defaults_exist_before_new_and_tim_sort() {
        let source = concat!(
            "var/global/datum/sort_instance/sorter = new /datum/sort_instance\n",
            "var/global/observed = 0\n",
            "/datum/sort_instance\n\tvar/list/runBases = list()\n\tvar/list/runLens = list()\n",
            "/datum/sort_instance/New()\n\trunBases.Add(1)\n",
            "/datum/sort_instance/proc/timSort()\n\trunBases.Cut()\n\trunLens.Cut()\n\treturn runBases.len + runLens.len\n",
            "/world/Genesis()\n\tglobal.observed = sorter.timSort()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse("\"a\" = (/turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n")
            .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
        execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("sort instance defaults should precede New and timSort");
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/observed"))
                .map(|variable| &variable.value),
            Some(&Value::number(0.0))
        );
    }

    #[test]
    fn runtime_new_links_inherited_call_and_nested_new_defaults_before_constructor() {
        let source = concat!(
            "var/global/observed = 0\n/proc/make_base()\n\treturn 3\n/proc/make_child()\n\treturn 8\n",
            "/datum/token\n/datum/base\n\tvar/x = make_base()\n\tvar/datum/token/token = new /datum/token\n",
            "/datum/base/child\n\tx = make_child()\n/datum/base/child/New()\n\tglobal.observed = x + istype(token, /datum/token)\n",
            "/world/Genesis()\n\tnew /datum/base/child\n/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse("\"a\" = (/turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n").unwrap();
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .unwrap();
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|v| v.path.ends_with("/observed"))
                .map(|v| &v.value),
            Some(&Value::number(9.0))
        );
    }

    #[test]
    fn compatibility_sweep_collects_lifecycle_failures_without_hiding_good_targets() {
        let source = concat!(
            "/world/New()\n\tspawn(1) return 0\n",
            "/atom/proc/New()\n\treturn 0\n",
            "/area/test\n/turf/test\n/obj/test\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");

        let sweep = sweep_lifecycle_compatibility(&compilation, &procedures, &index, &plan);

        assert!(sweep.targets >= 2);
        assert!(sweep.compatible >= 1);
        assert_eq!(sweep.issues.len(), 1);
        assert!(!sweep.issues[0].message.is_empty());
        assert_eq!(
            sweep.issues[0].locations[0].procedure_path,
            "/world/proc/New"
        );
        assert_eq!(sweep.issues[0].locations[0].source.path, "types.dm");
    }

    #[test]
    fn runtime_audit_collects_independent_map_failures_in_one_execution() {
        let source = concat!(
            "/area/test\n/turf/test\n",
            "/obj/first\n/obj/first/Initialize()\n\tvar/list/missing\n\treturn missing[1]\n",
            "/obj/second\n/obj/second/Initialize()\n\tvar/list/missing\n\treturn missing[2]\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/first, /turf/test, /area/test)\n",
            "\"b\" = (/obj/second, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\nab\n\"}\n",
        ))
        .expect("two-cell audit map should parse");
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("both failing Initialize bodies should link");
        let plan = build_initialization_plan(&runtime, &index, &world, "audit.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("audit world should allocate");

        let error = audit_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            &mut precompiled,
        )
        .expect_err("audit should return its grouped failure count");

        assert!(matches!(
            error,
            InitializationExecutionError::AuditFailures { failures: 2 }
        ));
    }

    #[test]
    fn host_slice_budget_reacts_quickly_and_recovers_gradually() {
        let mut budget = HostSliceBudget::new(100_000, 1_000, 100_000, Duration::from_millis(10));

        budget.observe(Duration::from_millis(11));
        assert_eq!(budget.steps(), 50_000);
        budget.observe(Duration::from_millis(20));
        assert_eq!(budget.steps(), 25_000);

        budget.observe(Duration::from_millis(5));
        assert_eq!(budget.steps(), 31_250);
        budget.observe(Duration::from_millis(8));
        assert_eq!(budget.steps(), 31_250);
    }

    #[test]
    fn host_slice_budget_never_leaves_configured_bounds() {
        let mut budget = HostSliceBudget::new(999_999, 1_000, 100_000, Duration::from_millis(10));
        assert_eq!(budget.steps(), 100_000);
        for _ in 0..16 {
            budget.observe(Duration::from_secs(1));
        }
        assert_eq!(budget.steps(), 1_000);
        for _ in 0..32 {
            budget.observe(Duration::ZERO);
        }
        assert_eq!(budget.steps(), 100_000);
    }
}
