//! Execution subsystem for lifecycle initialization, datum construction/deletion,
//! and compatibility sweeping.
//!
//! This module owns [`InitializationExecution`], [`InitializationExecutionError`],
//! the `execute_initialization_plan*` family, [`construct_datum`], [`delete_datum`],
//! and the `sweep_lifecycle_compatibility*` functions.

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_core::SourceSpan;
use dm_map::MapVariableAssignment;
use dm_runtime::RuntimeImage;
use dm_semantics::{ExecutableProcedures, ProcedureImplementationId, ProcedureRegistry};
use dm_value::{DatumId, FieldName, TypePath, Value};
use dm_vm::{ExecutionContext, Module, RuntimeError, execute_module_in_context};
use dm_world::{
    AtomCategory, WorldAllocation, WorldAllocationWorkKind, WorldCoordinate,
    materialize_world_map_state,
};

use crate::initialization_plan::{
    EventSubject, InitializationEvent, InitializationPlan, PlannedAtom,
};
use crate::lifecycle_index::{
    LifecycleCompatibilityIssue, LifecycleCompatibilityLocation, LifecycleCompatibilitySweep,
    LifecycleIndex, LifecycleKind, LifecycleResolution, LifecycleTarget,
};
use crate::precompile::PrecompiledLifecycle;
use crate::readiness::HeadlessReadinessProbe;
use crate::scheduler::{SchedulerDrain, SchedulerDrainLimits, drain_startup_scheduler};

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
        Some(&mut precompiled.persistent_state),
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
    startup_service: &mut dyn FnMut(&ExecutableProcedures, &mut dm_vm::ExecutionState),
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
        Some(&mut precompiled.persistent_state),
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
    executable: &mut ExecutableProcedures,
    persistent_state: Option<&mut Option<dm_vm::ExecutionState>>,
    collect_runtime_errors: bool,
    release_runtime_metadata: bool,
    startup_service: Option<&mut dyn FnMut(&ExecutableProcedures, &mut dm_vm::ExecutionState)>,
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
    module: &mut Module,
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

fn event_target<'a>(
    event: InitializationEvent,
    index: &'a LifecycleIndex,
) -> Option<&'a LifecycleTarget> {
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

// ── PrecompiledLifecycle runtime/execution methods ───────────────────────────

impl PrecompiledLifecycle {
    /// Installs a validated ready-world state loaded from the startup cache.
    #[doc(hidden)]
    pub fn install_persistent_state(&mut self, state: dm_vm::ExecutionState) {
        self.persistent_state = Some(state);
    }

    /// Returns the live VM state retained by a completed headless boot.
    ///
    /// Hosts use this at the persistent-scheduler boundary to service local
    /// client sessions without duplicating world state in a second runtime.
    #[must_use]
    pub fn persistent_state_mut(&mut self) -> Option<&mut dm_vm::ExecutionState> {
        self.persistent_state.as_mut()
    }

    /// Marks the startup/steady-state boundary for `DREAM64_PROFILE_INSTRUCTIONS`.
    /// The server calls this once, when it opens the lobby.
    pub fn mark_profile_steady_state(&mut self) {
        if let Some(state) = self.persistent_state.as_mut() {
            state.mark_profile_steady_state();
        }
    }

    /// `(gc_count, gc_ms, executed_steps)` accumulated by the live world, for
    /// the boot benchmark. `(0, 0, 0)` before the persistent world exists.
    #[must_use]
    pub fn boot_execution_totals(&self) -> (u64, u128, u64) {
        self.persistent_state.as_ref().map_or((0, 0, 0), |state| {
            let (count, elapsed) = state.list_gc_totals();
            (count, elapsed.as_millis(), state.total_executed_steps())
        })
    }

    /// Static-field quickening counters for every `obj.field` read. All zero
    /// before the persistent world exists. A low hit ratio here explains a
    /// large `field-read` profile line.
    #[must_use]
    pub fn field_quickening_totals(&self) -> dm_vm::DeclaredFieldQuickeningMetrics {
        self.persistent_state
            .as_ref()
            .map(dm_vm::ExecutionState::declared_field_quickening_metrics)
            .unwrap_or_default()
    }

    /// `(cache hits, cold ancestry walks)` for `effective_initial_value` — the
    /// resolver every unmaterialized field read goes through.
    #[must_use]
    pub fn effective_initial_value_totals(&self) -> (u64, u64) {
        self.persistent_state.as_ref().map_or(
            (0, 0),
            dm_vm::ExecutionState::effective_initial_value_totals,
        )
    }

    /// `DREAM64_PROFILE_INSTRUCTIONS` histogram lines for the phase
    /// (`false` = startup, `true` = steady-state). Empty unless enabled.
    #[must_use]
    pub fn instruction_profile_lines(&self, phase_steady: bool) -> Vec<String> {
        self.persistent_state
            .as_ref()
            .map_or_else(Vec::new, |state| {
                state.instruction_profile_lines(phase_steady)
            })
    }

    /// `DREAM64_PROFILE_PROC_STEPS` accounting: the `limit` procedures with the
    /// most self-time steps as `steps=N pct=P.P procedure=/path`. Empty unless set.
    #[must_use]
    pub fn proc_step_profile_lines(&self, limit: usize) -> Vec<String> {
        let module = self.executable.module();
        self.persistent_state
            .as_ref()
            .map_or_else(Vec::new, |state| {
                let total = state.total_executed_steps().max(1);
                state
                    .proc_step_profile_top(module, limit)
                    .into_iter()
                    .map(|(path, steps)| {
                        #[allow(clippy::cast_precision_loss)]
                        let pct = (steps as f64 / total as f64) * 100.0;
                        format!("steps={steps} pct={pct:.1} procedure={path}")
                    })
                    .collect()
            })
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
