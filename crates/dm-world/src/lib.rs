//! Deterministic, non-executing world-instantiation plans for parsed DMM maps.
//!
//! This layer resolves map initializer paths against a compiled DM object tree
//! and expands grid cells into world coordinates. It deliberately does not run
//! `New()`, `Initialize()`, or any other user code.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::BTreeMap;

use dm_compiler::Compilation;
use dm_core::SourceSpan;
use dm_map::{Map, MapVariableAssignment};
use dm_object_tree::{CodeTree, NodeId, NodeKind};

mod allocation;

pub use allocation::{
    CoordinateDatumSnapshot, WorldAllocation, WorldAllocationError, WorldAllocationStats,
    WorldAllocationWorkItem, WorldAllocationWorkKind, allocate_world,
};

/// One stable integer world coordinate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WorldCoordinate {
    /// X coordinate.
    pub x: i32,
    /// Y coordinate.
    pub y: i32,
    /// Z coordinate.
    pub z: i32,
}

/// Runtime-relevant category inferred through effective type inheritance.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum AtomCategory {
    /// An `/area` or descendant.
    Area,
    /// A `/turf` or descendant.
    Turf,
    /// An `/atom/movable` descendant, including objects and mobs.
    Movable,
    /// A valid type outside the three world-atom categories.
    OtherType,
}

/// Object-tree resolution result for one source initializer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitializerResolution {
    /// The path names an instantiable type.
    Resolved {
        /// Stable object-tree type identity.
        node: NodeId,
        /// Category derived from effective inheritance.
        category: AtomCategory,
    },
    /// No object-tree node has this absolute path.
    Unknown,
    /// The path exists but names a procedure, verb, or variable.
    NonType {
        /// Stable identity of the non-type node.
        node: NodeId,
        /// Namespace occupied by the node.
        kind: NodeKind,
    },
}

/// One source-ordered atom initializer prepared for later construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedInitializer {
    /// Absolute source type path.
    pub path: String,
    /// Complete map-source span of this initializer.
    pub span: SourceSpan,
    /// Ordered, lossless variable assignments to apply after allocation.
    pub variables: Vec<MapVariableAssignment>,
    /// Object-tree resolution state.
    pub resolution: InitializerResolution,
}

/// Reusable instantiation template for one map key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellTemplate {
    /// Quoted map key.
    pub key: String,
    /// Complete source span of the key definition.
    pub span: SourceSpan,
    /// Initializers in DMM creation order.
    pub initializers: Vec<PlannedInitializer>,
    /// Whether at least one resolved initializer derives from `/area`.
    pub has_area: bool,
    /// Whether at least one resolved initializer derives from `/turf`.
    pub has_turf: bool,
}

/// One coordinate expanded from a block row and column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedCell {
    /// Expanded stable world coordinate.
    pub coordinate: WorldCoordinate,
    /// Key selecting a [`CellTemplate`].
    pub key: String,
    /// Complete source span of the containing coordinate block.
    pub block_span: SourceSpan,
    /// Source span of the selected key definition, when it exists.
    pub template_span: Option<SourceSpan>,
}

/// Stable diagnostic category emitted during planning.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WorldDiagnosticKind {
    /// An initializer path has no object-tree node.
    UnknownTypePath,
    /// An initializer path exists in a non-type namespace.
    PathNotType,
    /// Two expanded cells occupy the same coordinate.
    DuplicateCoordinate,
    /// Coordinate expansion exceeded the signed 32-bit DMM coordinate range.
    CoordinateOverflow,
    /// A cell references a key absent from the supplied map table.
    MissingKeyDefinition,
    /// A cell template contains no resolved `/area` initializer.
    MissingArea,
    /// A cell template contains no resolved `/turf` initializer.
    MissingTurf,
}

impl WorldDiagnosticKind {
    /// Returns a stable report spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::UnknownTypePath => "unknown-type-path",
            Self::PathNotType => "path-not-type",
            Self::DuplicateCoordinate => "duplicate-coordinate",
            Self::CoordinateOverflow => "coordinate-overflow",
            Self::MissingKeyDefinition => "missing-key-definition",
            Self::MissingArea => "missing-area",
            Self::MissingTurf => "missing-turf",
        }
    }
}

/// One recoverable, source-mapped planning diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldDiagnostic {
    /// Stable diagnostic category.
    pub kind: WorldDiagnosticKind,
    /// Human-readable detail.
    pub message: String,
    /// Relevant map-source span.
    pub span: SourceSpan,
    /// Earlier related source span, such as the first overlapping block.
    pub previous_span: Option<SourceSpan>,
    /// Affected coordinate, when expansion produced one.
    pub coordinate: Option<WorldCoordinate>,
    /// Affected initializer path, when applicable.
    pub path: Option<String>,
}

/// Deterministic planning counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldPlanStats {
    /// Key templates retained from the map.
    pub templates: usize,
    /// Unique coordinates retained in the plan.
    pub cells: usize,
    /// Source atom initializers across key templates.
    pub initializers: usize,
    /// Initializers successfully resolved to type nodes.
    pub resolved_initializers: usize,
    /// All source initializer placements across expanded cells.
    pub initializer_placements: usize,
    /// Resolved initializer placements across expanded cells.
    pub resolved_atom_placements: usize,
    /// Recoverable planning diagnostics.
    pub diagnostics: usize,
}

/// A deterministic world-construction plan with no executed DM lifecycle code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldPlan {
    templates: BTreeMap<String, CellTemplate>,
    cells: Vec<PlannedCell>,
    diagnostics: Vec<WorldDiagnostic>,
    stats: WorldPlanStats,
}

impl WorldPlan {
    /// Returns key templates in stable lexical key order.
    #[must_use]
    pub const fn templates(&self) -> &BTreeMap<String, CellTemplate> {
        &self.templates
    }

    /// Returns expanded cells in coordinate-block, row, and column order.
    #[must_use]
    pub fn cells(&self) -> &[PlannedCell] {
        &self.cells
    }

    /// Returns diagnostics in deterministic planning encounter order.
    #[must_use]
    pub fn diagnostics(&self) -> &[WorldDiagnostic] {
        &self.diagnostics
    }

    /// Returns deterministic planning counters.
    #[must_use]
    pub const fn stats(&self) -> &WorldPlanStats {
        &self.stats
    }

    /// Finds the template selected by a planned cell key.
    #[must_use]
    pub fn template(&self, key: &str) -> Option<&CellTemplate> {
        self.templates.get(key)
    }
}

/// Builds a recoverable world-instantiation plan from a parsed map and object tree.
///
/// This operation performs no allocations of DM objects and invokes no DM code.
#[must_use]
pub fn build_plan(map: &Map, compilation: &Compilation) -> WorldPlan {
    let tree = compilation.code_tree();
    let paths: BTreeMap<_, _> = tree
        .nodes()
        .iter()
        .map(|node| (node.path.to_string(), (node.id, node.kind)))
        .collect();
    let roots = CategoryRoots {
        area: paths.get("/area").map(|entry| entry.0),
        turf: paths.get("/turf").map(|entry| entry.0),
        movable: paths.get("/atom/movable").map(|entry| entry.0),
    };
    let mut diagnostics = Vec::new();
    let templates = build_templates(map, tree, &paths, roots, &mut diagnostics);
    let cells = expand_cells(map, &templates, &mut diagnostics);
    let stats = plan_stats(&templates, &cells, diagnostics.len());
    WorldPlan {
        templates,
        cells,
        diagnostics,
        stats,
    }
}

fn build_templates(
    map: &Map,
    tree: &CodeTree,
    paths: &BTreeMap<String, (NodeId, NodeKind)>,
    roots: CategoryRoots,
    diagnostics: &mut Vec<WorldDiagnostic>,
) -> BTreeMap<String, CellTemplate> {
    map.keys
        .keys()
        .map(|key| {
            let definition = &map.keys[key];
            let initializers = definition
                .atoms
                .iter()
                .map(|initializer| {
                    let resolution = resolve_initializer(
                        tree,
                        paths,
                        roots,
                        &initializer.path,
                        initializer.span,
                        diagnostics,
                    );
                    PlannedInitializer {
                        path: initializer.path.clone(),
                        span: initializer.span,
                        variables: initializer.variable_assignments.clone(),
                        resolution,
                    }
                })
                .collect::<Vec<_>>();
            let has_area = has_category(&initializers, AtomCategory::Area);
            let has_turf = has_category(&initializers, AtomCategory::Turf);
            (
                key.clone(),
                CellTemplate {
                    key: key.clone(),
                    span: definition.span,
                    initializers,
                    has_area,
                    has_turf,
                },
            )
        })
        .collect()
}

fn expand_cells(
    map: &Map,
    templates: &BTreeMap<String, CellTemplate>,
    diagnostics: &mut Vec<WorldDiagnostic>,
) -> Vec<PlannedCell> {
    let mut cells = Vec::new();
    let mut occupied = BTreeMap::new();
    for block in &map.blocks {
        for (row_index, row) in block.rows.iter().enumerate() {
            for (column_index, key) in row.iter().enumerate() {
                let Some(coordinate) = expanded_coordinate(block, column_index, row_index) else {
                    diagnostics.push(WorldDiagnostic {
                        kind: WorldDiagnosticKind::CoordinateOverflow,
                        message: "map cell coordinate exceeds the signed 32-bit range".to_owned(),
                        span: block.span,
                        previous_span: None,
                        coordinate: None,
                        path: None,
                    });
                    continue;
                };
                if let Some(previous_span) = occupied.get(&coordinate).copied() {
                    diagnostics.push(WorldDiagnostic {
                        kind: WorldDiagnosticKind::DuplicateCoordinate,
                        message: format!(
                            "more than one map cell expands to ({},{},{})",
                            coordinate.x, coordinate.y, coordinate.z
                        ),
                        span: block.span,
                        previous_span: Some(previous_span),
                        coordinate: Some(coordinate),
                        path: None,
                    });
                    continue;
                }
                occupied.insert(coordinate, block.span);
                let template = templates.get(key);
                cells.push(PlannedCell {
                    coordinate,
                    key: key.clone(),
                    block_span: block.span,
                    template_span: template.map(|template| template.span),
                });
                diagnose_cell_structure(template, key, coordinate, block.span, diagnostics);
            }
        }
    }
    cells
}

fn plan_stats(
    templates: &BTreeMap<String, CellTemplate>,
    cells: &[PlannedCell],
    diagnostics: usize,
) -> WorldPlanStats {
    let initializers = templates
        .values()
        .map(|template| template.initializers.len())
        .sum();
    let resolved_initializers = templates
        .values()
        .flat_map(|template| &template.initializers)
        .filter(|initializer| {
            matches!(
                initializer.resolution,
                InitializerResolution::Resolved { .. }
            )
        })
        .count();
    let initializer_placements = cells
        .iter()
        .filter_map(|cell| templates.get(&cell.key))
        .map(|template| template.initializers.len())
        .sum();
    let resolved_atom_placements = cells
        .iter()
        .filter_map(|cell| templates.get(&cell.key))
        .map(|template| {
            template
                .initializers
                .iter()
                .filter(|initializer| {
                    matches!(
                        initializer.resolution,
                        InitializerResolution::Resolved { .. }
                    )
                })
                .count()
        })
        .sum();
    WorldPlanStats {
        templates: templates.len(),
        cells: cells.len(),
        initializers,
        resolved_initializers,
        initializer_placements,
        resolved_atom_placements,
        diagnostics,
    }
}

#[derive(Clone, Copy)]
struct CategoryRoots {
    area: Option<NodeId>,
    turf: Option<NodeId>,
    movable: Option<NodeId>,
}

fn resolve_initializer(
    tree: &CodeTree,
    paths: &BTreeMap<String, (NodeId, NodeKind)>,
    roots: CategoryRoots,
    path: &str,
    span: SourceSpan,
    diagnostics: &mut Vec<WorldDiagnostic>,
) -> InitializerResolution {
    let Some((node, kind)) = paths.get(path).copied() else {
        diagnostics.push(WorldDiagnostic {
            kind: WorldDiagnosticKind::UnknownTypePath,
            message: format!("map initializer path {path:?} is absent from the object tree"),
            span,
            previous_span: None,
            coordinate: None,
            path: Some(path.to_owned()),
        });
        return InitializerResolution::Unknown;
    };
    if kind != NodeKind::Type {
        diagnostics.push(WorldDiagnostic {
            kind: WorldDiagnosticKind::PathNotType,
            message: format!("map initializer path {path:?} names a {kind:?}, not a type"),
            span,
            previous_span: None,
            coordinate: None,
            path: Some(path.to_owned()),
        });
        return InitializerResolution::NonType { node, kind };
    }
    InitializerResolution::Resolved {
        node,
        category: category(tree, node, roots),
    }
}

fn category(tree: &CodeTree, node: NodeId, roots: CategoryRoots) -> AtomCategory {
    if roots.area.is_some_and(|root| inherits(tree, node, root)) {
        AtomCategory::Area
    } else if roots.turf.is_some_and(|root| inherits(tree, node, root)) {
        AtomCategory::Turf
    } else if roots.movable.is_some_and(|root| inherits(tree, node, root)) {
        AtomCategory::Movable
    } else {
        AtomCategory::OtherType
    }
}

fn inherits(tree: &CodeTree, mut node: NodeId, ancestor: NodeId) -> bool {
    for _ in 0..=tree.nodes().len() {
        if node == ancestor {
            return true;
        }
        let Some(parent) = tree.node(node).and_then(|current| current.parent_type) else {
            return false;
        };
        node = parent;
    }
    false
}

fn has_category(initializers: &[PlannedInitializer], expected: AtomCategory) -> bool {
    initializers.iter().any(|initializer| {
        matches!(
            initializer.resolution,
            InitializerResolution::Resolved { category, .. } if category == expected
        )
    })
}

fn expanded_coordinate(
    block: &dm_map::MapBlock,
    column_index: usize,
    row_index: usize,
) -> Option<WorldCoordinate> {
    let column = i32::try_from(column_index).ok()?;
    let row = i32::try_from(row_index).ok()?;
    Some(WorldCoordinate {
        x: block.x.checked_add(column)?,
        y: block.y.checked_add(row)?,
        z: block.z,
    })
}

fn diagnose_cell_structure(
    template: Option<&CellTemplate>,
    key: &str,
    coordinate: WorldCoordinate,
    block_span: SourceSpan,
    diagnostics: &mut Vec<WorldDiagnostic>,
) {
    let Some(template) = template else {
        diagnostics.push(WorldDiagnostic {
            kind: WorldDiagnosticKind::MissingKeyDefinition,
            message: format!("map cell references missing key {key:?}"),
            span: block_span,
            previous_span: None,
            coordinate: Some(coordinate),
            path: None,
        });
        return;
    };
    if !template.has_area {
        diagnostics.push(WorldDiagnostic {
            kind: WorldDiagnosticKind::MissingArea,
            message: format!("map cell key {key:?} has no resolved area"),
            span: template.span,
            previous_span: None,
            coordinate: Some(coordinate),
            path: None,
        });
    }
    if !template.has_turf {
        diagnostics.push(WorldDiagnostic {
            kind: WorldDiagnosticKind::MissingTurf,
            message: format!("map cell key {key:?} has no resolved turf"),
            span: template.span,
            previous_span: None,
            coordinate: Some(coordinate),
            path: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::{Compilation, CompilerDatabase};
    use dm_map::parse;

    use super::{
        AtomCategory, InitializerResolution, WorldCoordinate, WorldDiagnosticKind, build_plan,
    };

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct TestProject {
        root: PathBuf,
    }

    impl TestProject {
        fn compile(source: &str) -> (Self, Compilation) {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("dream64-dm-world-{}-{ordinal}", std::process::id()));
            fs::create_dir(&root).expect("test project directory should be created");
            fs::write(root.join("world.dme"), "#include \"types.dm\"\n")
                .expect("environment should be written");
            fs::write(root.join("types.dm"), source).expect("types should be written");
            let project = Self { root };
            let compilation = CompilerDatabase::new()
                .compile(project.root.join("world.dme"))
                .expect("test project should compile");
            (project, compilation)
        }
    }

    impl Drop for TestProject {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn compilation() -> (TestProject, Compilation) {
        TestProject::compile(
            "/area/test\n/turf/open/test\n/obj/item/test\n\tproc/not_a_type()\n\t\treturn\n",
        )
    }

    #[test]
    fn resolves_templates_and_expands_stable_coordinates_without_execution() {
        let (_project, compilation) = compilation();
        let source = concat!(
            "\"a\" = (/obj/item/test{name = \"first\"; dir = 4}, /turf/open/test, /area/test)\n",
            "(5,7,2) = {\"\naa\na\n\"}\n",
        );
        let map = parse(source).expect("map should parse");
        let plan = build_plan(&map, &compilation);

        assert!(plan.diagnostics().is_empty());
        assert_eq!(
            plan.cells()
                .iter()
                .map(|cell| cell.coordinate)
                .collect::<Vec<_>>(),
            [
                WorldCoordinate { x: 5, y: 7, z: 2 },
                WorldCoordinate { x: 6, y: 7, z: 2 },
                WorldCoordinate { x: 5, y: 8, z: 2 },
            ]
        );
        let template = plan.template("a").expect("template should exist");
        assert_eq!(template.initializers[0].variables[0].name, "name");
        assert_eq!(template.initializers[0].variables[1].name, "dir");
        assert_eq!(
            template.initializers[0].resolution,
            InitializerResolution::Resolved {
                node: match template.initializers[0].resolution {
                    InitializerResolution::Resolved { node, .. } => node,
                    _ => panic!("movable should resolve"),
                },
                category: AtomCategory::Movable,
            }
        );
        assert_eq!(plan.stats().initializer_placements, 9);
        assert_eq!(plan.stats().resolved_atom_placements, 9);
    }

    #[test]
    fn diagnoses_unknown_non_type_and_incomplete_cells() {
        let (_project, compilation) = compilation();
        let source = concat!(
            "\"a\" = (/missing/type, /obj/item/test/proc/not_a_type, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        );
        let map = parse(source).expect("map should parse");
        let plan = build_plan(&map, &compilation);
        let kinds: Vec<_> = plan
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.kind)
            .collect();

        assert_eq!(
            kinds,
            [
                WorldDiagnosticKind::UnknownTypePath,
                WorldDiagnosticKind::PathNotType,
                WorldDiagnosticKind::MissingTurf,
            ]
        );
        assert!(
            plan.diagnostics()
                .iter()
                .all(|diagnostic| !diagnostic.span.is_empty())
        );
    }

    #[test]
    fn retains_first_cell_and_diagnoses_duplicate_coordinates() {
        let (_project, compilation) = compilation();
        let source = concat!(
            "\"a\" = (/turf/open/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
            "(1,1,1) = {\"\na\n\"}\n",
        );
        let map = parse(source).expect("map should parse");
        let plan = build_plan(&map, &compilation);

        assert_eq!(plan.cells().len(), 1);
        assert_eq!(plan.diagnostics().len(), 1);
        assert_eq!(
            plan.diagnostics()[0].kind,
            WorldDiagnosticKind::DuplicateCoordinate
        );
        assert!(plan.diagnostics()[0].previous_span.is_some());
    }
}
