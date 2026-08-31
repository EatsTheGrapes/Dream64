/// Deterministic, non-executing initialization planning.
///
/// Owns the GlobalInitialization/MapPlacementContext/PlannedAtom/
/// InitializationPlan types and the build_initialization_plan pipeline.
/// This is the initialization planning boundary in docs/ARCHITECTURE.md
/// and must remain independent of dm_vm::ExecutionState and
/// execution/scheduler machinery. Dependency direction is
/// LifecycleIndex -> InitializationPlan -> Execution.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use dm_core::SourceSpan;
use dm_map::MapVariableAssignment;
use dm_runtime::RuntimeImage;
use dm_world::{AtomCategory, InitializerResolution, WorldCoordinate, WorldPlan};

use crate::lifecycle_index::{
    LifecycleDiagnostic, LifecycleDiagnosticKind, LifecycleIndex, LifecycleKind,
    LifecycleResolution,
};
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
            let category = match initializer.resolution {
                InitializerResolution::Resolved { category } => category,
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
            let Some(type_index) = index.by_path.get(&initializer.path).copied() else {
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

pub(crate) fn project_manages_atom_initialization(index: &LifecycleIndex) -> bool {
    index
        .types
        .iter()
        .enumerate()
        .filter(|&(type_index, _)| inherits_from_subsystem_atoms(index, type_index))
        .any(|(type_index, _)| {
            matches!(
                index.types[type_index]
                    .targets
                    .get(LifecycleKind::Initialize),
                LifecycleResolution::Resolved(_)
            ) || matches!(
                index.types[type_index]
                    .targets
                    .get(LifecycleKind::LateInitialize),
                LifecycleResolution::Resolved(_)
            )
        })
}

fn inherits_from_subsystem_atoms(index: &LifecycleIndex, type_index: usize) -> bool {
    let mut current = type_index;
    let mut seen = BTreeSet::<usize>::new();
    loop {
        if !seen.insert(current) {
            return false;
        }
        let lifecycle = &index.types[current];
        if lifecycle.path == "/datum/controller/subsystem/atoms" {
            return true;
        }
        let Some(parent) = lifecycle.parent.as_deref() else {
            return false;
        };
        let Some(parent_index) = index.by_path.get(parent) else {
            return false;
        };
        current = *parent_index;
    }
}

fn has_target(index: &LifecycleIndex, type_index: usize, kind: LifecycleKind) -> bool {
    index.types.get(type_index).is_some_and(|lifecycle| {
        matches!(
            lifecycle.targets.get(kind),
            LifecycleResolution::Resolved(_)
        )
    })
}
