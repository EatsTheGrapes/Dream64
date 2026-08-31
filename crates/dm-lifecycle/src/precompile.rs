/// Precompilation pipeline for lifecycle bytecode.
///
/// Owns PrecompiledLifecycle construction and the lifecycle_targets_for_world
/// selection that links the exact world/map roots without allocating. This
/// module depends on LifecycleIndex, WorldPlan, and
/// initialization_plan::project_manages_atom_initialization for map-kind
/// selection. PrecompiledLifecycle still carries persistent_state:
/// Option<ExecutionState> — a hidden execution coupling that ideally would be
/// split into a separate PersistentLifecycle in a future execution extraction.
use std::collections::BTreeSet;

use dm_compiler::Compilation;
use dm_semantics::{ProcedureImplementationId, ProcedureRegistry};
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
