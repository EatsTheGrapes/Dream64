/// Lifecycle resolution and indexing.
///
/// Owns the portable D64LIDX directory codec and the effective
/// Genesis/New/Initialize/LateInitialize/Destroy dispatch table for
/// every canonical runtime type. This is the lifecycle resolution/indexing
/// boundary described in docs/ARCHITECTURE.md and must remain independent
/// of initialization planning, execution/scheduling, readiness, and map
/// catalog products.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use dm_compiler::Compilation;
use dm_core::{FileId, SourceSpan};
use dm_object_tree::{NodeId, NodeKind};
use dm_runtime::RuntimeImage;
use dm_semantics::{Procedure, ProcedureId, ProcedureImplementationId, ProcedureRegistry};
use dm_world::WorldCoordinate;
use serde::{Deserialize, Serialize};
const LIFECYCLE_DIRECTORY_MAGIC: &[u8; 8] = b"D64LIDX\0";
const LIFECYCLE_DIRECTORY_VERSION: u16 = 1;
const MAX_LIFECYCLE_DIRECTORY_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct PortableLifecycleDirectory {
    types: Vec<PortableTypeLifecycle>,
}

#[derive(Serialize, Deserialize)]
struct PortableTypeLifecycle {
    path: String,
    parent: Option<String>,
    targets: [PortableLifecycleResolution; 5],
}

#[derive(Serialize, Deserialize)]
enum PortableLifecycleResolution {
    Absent,
    Resolved {
        procedure: u32,
        implementation: u32,
        procedure_path: String,
        declaring_type: String,
        inherited: bool,
        source_path: String,
        source_start: u64,
        source_end: u64,
        source_ordinal: u64,
    },
    Unsupported {
        message: String,
        procedure_path: Option<String>,
    },
}

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
    /// All lifecycle kinds in canonical order.
    pub const ALL: [Self; 5] = [
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
    pub(crate) types: Vec<TypeLifecycle>,
    pub(crate) by_node: BTreeMap<NodeId, usize>,
    pub(crate) by_path: BTreeMap<String, usize>,
    pub(crate) diagnostics: Vec<LifecycleDiagnostic>,
}

fn portable_resolution(value: &LifecycleResolution) -> PortableLifecycleResolution {
    match value {
        LifecycleResolution::Absent => PortableLifecycleResolution::Absent,
        LifecycleResolution::Resolved(target) => PortableLifecycleResolution::Resolved {
            procedure: target.procedure.index() as u32,
            implementation: target.implementation.index() as u32,
            procedure_path: target.procedure_path.clone(),
            declaring_type: target.declaring_type.clone(),
            inherited: target.inherited,
            source_path: target.source.path.clone(),
            source_start: target.source.span.start as u64,
            source_end: target.source.span.end as u64,
            source_ordinal: target.source.ordinal as u64,
        },
        LifecycleResolution::Unsupported(issue) => PortableLifecycleResolution::Unsupported {
            message: issue.message.clone(),
            procedure_path: issue.procedure_path.clone(),
        },
    }
}

fn runtime_resolution(value: PortableLifecycleResolution) -> LifecycleResolution {
    match value {
        PortableLifecycleResolution::Absent => LifecycleResolution::Absent,
        PortableLifecycleResolution::Unsupported {
            message,
            procedure_path,
        } => LifecycleResolution::Unsupported(Arc::new(LifecycleTargetIssue {
            kind: LifecycleTargetIssueKind::MissingImplementation,
            message,
            procedure_path,
        })),
        PortableLifecycleResolution::Resolved {
            procedure,
            implementation,
            procedure_path,
            declaring_type,
            inherited,
            source_path,
            source_start,
            source_end,
            source_ordinal,
        } => {
            let procedure = ProcedureId::from_index(procedure as usize);
            LifecycleResolution::Resolved(Arc::new(LifecycleTarget {
                procedure,
                implementation: ProcedureImplementationId::from_indices(
                    procedure.index(),
                    implementation as usize,
                ),
                procedure_path,
                declaring_type,
                inherited,
                source: LifecycleSource {
                    file_id: FileId::from_index(0),
                    path: source_path,
                    span: SourceSpan::new(source_start as usize, source_end as usize),
                    ordinal: source_ordinal as usize,
                },
            }))
        }
    }
}

impl LifecycleIndex {
    /// Encodes the runtime lifecycle directory without compiler-local node identities.
    pub fn encode_portable(&self) -> Result<Vec<u8>, String> {
        use bincode::Options as _;
        let directory = PortableLifecycleDirectory {
            types: self
                .types
                .iter()
                .map(|ty| PortableTypeLifecycle {
                    path: ty.path.clone(),
                    parent: ty.parent.clone(),
                    targets: LifecycleKind::ALL
                        .map(|kind| portable_resolution(ty.targets.get(kind))),
                })
                .collect(),
        };
        let payload = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(MAX_LIFECYCLE_DIRECTORY_BYTES)
            .serialize(&directory)
            .map_err(|error| error.to_string())?;
        let mut bytes = Vec::with_capacity(22 + payload.len());
        bytes.extend_from_slice(LIFECYCLE_DIRECTORY_MAGIC);
        bytes.extend_from_slice(&LIFECYCLE_DIRECTORY_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    /// Decodes a portable lifecycle directory and assigns process-local node handles.
    pub fn decode_portable(bytes: &[u8]) -> Result<Self, String> {
        use bincode::Options as _;
        if bytes.len() < 22 || &bytes[..8] != LIFECYCLE_DIRECTORY_MAGIC {
            return Err("bad lifecycle directory header".into());
        }
        if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != LIFECYCLE_DIRECTORY_VERSION {
            return Err("unsupported lifecycle directory version".into());
        }
        let length = usize::try_from(u64::from_le_bytes(bytes[10..18].try_into().unwrap()))
            .map_err(|_| "lifecycle directory length exceeds usize")?;
        if length as u64 > MAX_LIFECYCLE_DIRECTORY_BYTES
            || bytes.len()
                != 22usize
                    .checked_add(length)
                    .ok_or("lifecycle directory length overflow")?
        {
            return Err("invalid lifecycle directory length".into());
        }
        let payload = &bytes[22..];
        if crc32fast::hash(payload) != u32::from_le_bytes(bytes[18..22].try_into().unwrap()) {
            return Err("lifecycle directory checksum mismatch".into());
        }
        let directory: PortableLifecycleDirectory = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(MAX_LIFECYCLE_DIRECTORY_BYTES)
            .reject_trailing_bytes()
            .deserialize(payload)
            .map_err(|error| error.to_string())?;
        let types = directory
            .types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| {
                let [genesis, new_target, initialize, late_initialize, destroy] =
                    ty.targets.map(runtime_resolution);
                TypeLifecycle {
                    node: NodeId::from_index(index),
                    path: ty.path,
                    parent: ty.parent,
                    targets: LifecycleTargets {
                        genesis,
                        new_target,
                        initialize,
                        late_initialize,
                        destroy,
                    },
                }
            })
            .collect::<Vec<_>>();
        let by_node = types
            .iter()
            .enumerate()
            .map(|(index, ty)| (ty.node, index))
            .collect();
        let by_path = types
            .iter()
            .enumerate()
            .map(|(index, ty)| (ty.path.clone(), index))
            .collect();
        Ok(Self {
            types,
            by_node,
            by_path,
            diagnostics: Vec::new(),
        })
    }

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
