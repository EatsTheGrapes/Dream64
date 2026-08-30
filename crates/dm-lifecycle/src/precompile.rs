/// Precompilation pipeline for lifecycle bytecode.
///
/// Owns PrecompiledLifecycle construction and the lifecycle_targets_for_world
/// selection that links the exact world/map roots without allocating. This
/// module depends on LifecycleIndex, WorldPlan, and
/// initialization_plan::project_manages_atom_initialization for map-kind
/// selection. PrecompiledLifecycle still carries persistent_state:
/// Option<ExecutionState> — a hidden execution coupling that ideally would be
/// split into a separate PersistentLifecycle in a future execution extraction.
/// For now the field and its ExecutionState-touching methods remain here to
/// preserve exact behavior and API compatibility; precompile.rs therefore
/// still imports dm_vm::ExecutionState (see remaining-concerns note).
use std::collections::BTreeSet;
use std::time::Duration;

use dm_compiler::Compilation;
use dm_semantics::{ExecutableProcedures, ProcedureImplementationId, ProcedureRegistry};
use dm_value::{DatumId, FieldName, Value};
use dm_vm::{CompileError, ExecutionState, LocalClientState, LocalScreenPointerEvent, Module};
use dm_world::{InitializerResolution, WorldPlan};

use crate::lifecycle_index::{LifecycleIndex, LifecycleKind, LifecycleResolution};

/// Lifecycle bytecode linked before runtime/world materialization so boot does
/// not overlap its closure/spec peak with the resident runtime heap.
pub struct PrecompiledLifecycle {
    pub(crate) executable: dm_semantics::ExecutableProcedures,
    pub(crate) persistent_state: Option<dm_vm::ExecutionState>,
    pub(crate) targets: usize,
    pub(crate) reachable_bodies: usize,
    pub(crate) closure: dm_semantics::ProcedureClosureStats,
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

/// Prepares lifecycle execution from portable runtime directories only.
#[must_use]
pub fn precompile_portable_lifecycle_for_world(
    index: &LifecycleIndex,
    world: &WorldPlan,
    executable: dm_semantics::ExecutableProcedures,
) -> PrecompiledLifecycle {
    let targets = lifecycle_targets_for_world(index, world);
    let reachable_bodies = executable.stats().procedures;
    PrecompiledLifecycle {
        executable,
        persistent_state: None,
        targets: targets.len(),
        reachable_bodies,
        closure: dm_semantics::ProcedureClosureStats::default(),
    }
}

pub(crate) fn lifecycle_targets_for_world(
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
    let map_kinds: &[LifecycleKind] =
        if crate::initialization_plan::project_manages_atom_initialization(index) {
            &[LifecycleKind::New]
        } else {
            &[
                LifecycleKind::New,
                LifecycleKind::Initialize,
                LifecycleKind::LateInitialize,
            ]
        };
    for path in world
        .templates()
        .values()
        .flat_map(|template| &template.initializers)
        .filter_map(|initializer| match initializer.resolution {
            InitializerResolution::Resolved { .. } => Some(initializer.path.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
    {
        let Some(lifecycle) = index.find_path(path) else {
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

pub(crate) fn precompiled_lifecycle_from_executable(
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
