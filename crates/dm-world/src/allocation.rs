use std::collections::BTreeMap;
use std::fmt;

use dm_core::SourceSpan;
use dm_globals::UnsupportedCategory;
use dm_runtime::{ConstantFieldApplication, RuntimeImage, RuntimeImageError};
use dm_value::{DatumId, FieldName, TypePath, ValueError};

use crate::{
    AtomCategory, CellTemplate, InitializerResolution, PlannedInitializer, WorldCoordinate,
    WorldPlan,
};

/// Inert runtime datum identities associated with one planned coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoordinateDatumSnapshot {
    /// Expanded world coordinate.
    pub coordinate: WorldCoordinate,
    /// Shared area identity selected by this coordinate.
    pub area: Option<DatumId>,
    /// Unique turf identity selected by this coordinate.
    pub turf: Option<DatumId>,
    /// Unique movable identities in map initializer order.
    pub movables: Vec<DatumId>,
    /// Every retained initializer identity in original map source order.
    pub source_order: Vec<DatumId>,
}

/// Recoverable allocation work that requires later semantics or execution.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WorldAllocationWorkKind {
    /// The plan could not resolve an initializer path.
    UnknownInitializer,
    /// The initializer path names a non-type object-tree node.
    NonTypeInitializer,
    /// A valid non-atom type is outside inert world allocation scope.
    OtherType,
    /// A coordinate refers to a missing key template.
    MissingTemplate,
    /// A cell contains more than one area initializer.
    ExtraArea,
    /// A cell contains more than one turf initializer.
    ExtraTurf,
    /// A map assignment target is not a valid runtime field name.
    InvalidFieldName,
    /// An override requires runtime evaluation or unsupported semantics.
    DynamicOverride(UnsupportedCategory),
}

/// One source-mapped item deliberately deferred by inert allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldAllocationWorkItem {
    /// Conservative reason the item was deferred.
    pub kind: WorldAllocationWorkKind,
    /// Coordinate whose source initializer requested the work.
    pub coordinate: WorldCoordinate,
    /// Complete source span of the initializer or missing template reference.
    pub span: SourceSpan,
    /// Canonical initializer path, when applicable.
    pub initializer_path: Option<String>,
    /// Map field target, when this item concerns an override.
    pub field: Option<String>,
    /// Complete assignment span, when this item concerns an override.
    pub assignment_span: Option<SourceSpan>,
    /// Precise unsupported expression span, when available.
    pub blocker_span: Option<SourceSpan>,
    /// Lossless raw map value retained for later evaluation.
    pub raw_value: Option<String>,
}

/// Deterministic inert-allocation counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldAllocationStats {
    /// Coordinate snapshots produced.
    pub cells: usize,
    /// Distinct runtime datums allocated.
    pub datums_allocated: usize,
    /// Shared area instances allocated from unique initializer signatures.
    pub unique_areas: usize,
    /// Unique turf instances allocated.
    pub turfs: usize,
    /// Unique movable instances allocated.
    pub movables: usize,
    /// Proven constant map assignments applied.
    pub constant_overrides: usize,
    /// Map assignments retained for later runtime evaluation.
    pub unsupported_overrides: usize,
    /// Initializers skipped because they were unresolved, extra, or non-atom.
    pub skipped_initializers: usize,
}

/// Deterministic result of allocating a [`WorldPlan`] without lifecycle calls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldAllocation {
    snapshots: Vec<CoordinateDatumSnapshot>,
    allocation_order: Vec<DatumId>,
    work_items: Vec<WorldAllocationWorkItem>,
    stats: WorldAllocationStats,
}

impl WorldAllocation {
    /// Returns coordinate snapshots in plan source order.
    #[must_use]
    pub fn snapshots(&self) -> &[CoordinateDatumSnapshot] {
        &self.snapshots
    }

    /// Looks up one coordinate snapshot.
    #[must_use]
    pub fn coordinate(&self, coordinate: WorldCoordinate) -> Option<&CoordinateDatumSnapshot> {
        self.snapshots
            .iter()
            .find(|snapshot| snapshot.coordinate == coordinate)
    }

    /// Returns distinct datum identities in actual heap allocation order.
    #[must_use]
    pub fn allocation_order(&self) -> &[DatumId] {
        &self.allocation_order
    }

    /// Returns deferred work in deterministic encounter order.
    #[must_use]
    pub fn work_items(&self) -> &[WorldAllocationWorkItem] {
        &self.work_items
    }

    /// Returns deterministic allocation counters.
    #[must_use]
    pub const fn stats(&self) -> &WorldAllocationStats {
        &self.stats
    }
}

/// Materializes a world plan into an existing runtime image without DM calls.
///
/// Areas are shared across cells when their canonical path and ordered raw map
/// assignments are identical. This models one map-defined area instance while
/// keeping differently overridden instances separate. Turfs and movables are
/// always unique per source initializer placement.
///
/// # Errors
///
/// Returns [`WorldAllocationError`] when the plan and runtime image disagree
/// about a resolved type or a heap operation fails.
pub fn allocate_world(
    plan: &WorldPlan,
    image: &mut RuntimeImage,
) -> Result<WorldAllocation, WorldAllocationError> {
    Allocator::new(plan, image).run()
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AreaInstanceKey {
    path: String,
    variables: Vec<(String, String)>,
}

impl AreaInstanceKey {
    fn new(initializer: &PlannedInitializer) -> Self {
        Self {
            path: initializer.path.clone(),
            variables: initializer
                .variables
                .iter()
                .map(|assignment| (assignment.name.clone(), assignment.value.raw.clone()))
                .collect(),
        }
    }
}

struct Allocator<'plan, 'image> {
    plan: &'plan WorldPlan,
    image: &'image mut RuntimeImage,
    areas: BTreeMap<AreaInstanceKey, DatumId>,
    snapshots: Vec<CoordinateDatumSnapshot>,
    allocation_order: Vec<DatumId>,
    work_items: Vec<WorldAllocationWorkItem>,
    stats: WorldAllocationStats,
}

impl<'plan, 'image> Allocator<'plan, 'image> {
    fn new(plan: &'plan WorldPlan, image: &'image mut RuntimeImage) -> Self {
        Self {
            plan,
            image,
            areas: BTreeMap::new(),
            snapshots: Vec::with_capacity(plan.cells().len()),
            allocation_order: Vec::new(),
            work_items: Vec::new(),
            stats: WorldAllocationStats::default(),
        }
    }

    fn run(mut self) -> Result<WorldAllocation, WorldAllocationError> {
        for cell in self.plan.cells() {
            let mut snapshot = CoordinateDatumSnapshot {
                coordinate: cell.coordinate,
                area: None,
                turf: None,
                movables: Vec::new(),
                source_order: Vec::new(),
            };
            let Some(template) = self.plan.template(&cell.key) else {
                self.work_items.push(WorldAllocationWorkItem {
                    kind: WorldAllocationWorkKind::MissingTemplate,
                    coordinate: cell.coordinate,
                    span: cell.block_span,
                    initializer_path: None,
                    field: None,
                    assignment_span: None,
                    blocker_span: None,
                    raw_value: None,
                });
                self.snapshots.push(snapshot);
                continue;
            };
            self.allocate_template(template, &mut snapshot)?;
            self.snapshots.push(snapshot);
        }
        self.stats.cells = self.snapshots.len();
        self.stats.datums_allocated = self.allocation_order.len();
        Ok(WorldAllocation {
            snapshots: self.snapshots,
            allocation_order: self.allocation_order,
            work_items: self.work_items,
            stats: self.stats,
        })
    }

    fn allocate_template(
        &mut self,
        template: &CellTemplate,
        snapshot: &mut CoordinateDatumSnapshot,
    ) -> Result<(), WorldAllocationError> {
        for initializer in &template.initializers {
            match initializer.resolution {
                InitializerResolution::Resolved {
                    category: AtomCategory::Area,
                    ..
                } if snapshot.area.is_some() => {
                    self.skip_initializer(
                        snapshot.coordinate,
                        initializer,
                        WorldAllocationWorkKind::ExtraArea,
                    );
                }
                InitializerResolution::Resolved {
                    category: AtomCategory::Turf,
                    ..
                } if snapshot.turf.is_some() => {
                    self.skip_initializer(
                        snapshot.coordinate,
                        initializer,
                        WorldAllocationWorkKind::ExtraTurf,
                    );
                }
                InitializerResolution::Resolved { category, .. } => {
                    self.allocate_resolved(initializer, category, snapshot)?;
                }
                InitializerResolution::Unknown => self.skip_initializer(
                    snapshot.coordinate,
                    initializer,
                    WorldAllocationWorkKind::UnknownInitializer,
                ),
                InitializerResolution::NonType { .. } => self.skip_initializer(
                    snapshot.coordinate,
                    initializer,
                    WorldAllocationWorkKind::NonTypeInitializer,
                ),
            }
        }
        Ok(())
    }

    fn allocate_resolved(
        &mut self,
        initializer: &PlannedInitializer,
        category: AtomCategory,
        snapshot: &mut CoordinateDatumSnapshot,
    ) -> Result<(), WorldAllocationError> {
        if category == AtomCategory::OtherType {
            self.skip_initializer(
                snapshot.coordinate,
                initializer,
                WorldAllocationWorkKind::OtherType,
            );
            return Ok(());
        }
        let datum = if category == AtomCategory::Area {
            let key = AreaInstanceKey::new(initializer);
            if let Some(datum) = self.areas.get(&key).copied() {
                datum
            } else {
                let datum = self.allocate_initializer(snapshot.coordinate, initializer)?;
                self.areas.insert(key, datum);
                self.stats.unique_areas += 1;
                datum
            }
        } else {
            self.allocate_initializer(snapshot.coordinate, initializer)?
        };
        snapshot.source_order.push(datum);
        match category {
            AtomCategory::Area => snapshot.area = Some(datum),
            AtomCategory::Turf => {
                snapshot.turf = Some(datum);
                self.stats.turfs += 1;
            }
            AtomCategory::Movable => {
                snapshot.movables.push(datum);
                self.stats.movables += 1;
            }
            AtomCategory::OtherType => {}
        }
        Ok(())
    }

    fn allocate_initializer(
        &mut self,
        coordinate: WorldCoordinate,
        initializer: &PlannedInitializer,
    ) -> Result<DatumId, WorldAllocationError> {
        let type_path = TypePath::parse(&initializer.path)?;
        let datum = self.image.allocate_datum(&type_path)?;
        // BYOND exposes map placement coordinates as built-in atom fields.
        // They are not map-variable overrides, so materialize them before
        // applying source-defined overrides and before lifecycle code runs.
        for (name, value) in [
            ("x", coordinate.x),
            ("y", coordinate.y),
            ("z", coordinate.z),
        ] {
            self.image.heap_mut().set_datum_field(
                datum,
                FieldName::parse(name).expect("coordinate field name is valid"),
                dm_value::Value::number(
                    value
                        .to_string()
                        .parse::<f32>()
                        .expect("world coordinate is representable as a DM number"),
                ),
            )?;
        }
        self.allocation_order.push(datum);
        for assignment in &initializer.variables {
            let Ok(field) = FieldName::parse(&assignment.name) else {
                self.stats.unsupported_overrides += 1;
                self.work_items.push(WorldAllocationWorkItem {
                    kind: WorldAllocationWorkKind::InvalidFieldName,
                    coordinate,
                    span: initializer.span,
                    initializer_path: Some(initializer.path.clone()),
                    field: Some(assignment.name.clone()),
                    assignment_span: Some(assignment.span),
                    blocker_span: Some(assignment.name_span),
                    raw_value: Some(assignment.value.raw.clone()),
                });
                continue;
            };
            match self
                .image
                .apply_constant_field_expression(datum, field, &assignment.value.raw)?
            {
                ConstantFieldApplication::Applied => self.stats.constant_overrides += 1,
                ConstantFieldApplication::Unsupported(unsupported) => {
                    self.stats.unsupported_overrides += 1;
                    self.work_items.push(WorldAllocationWorkItem {
                        kind: WorldAllocationWorkKind::DynamicOverride(unsupported.category),
                        coordinate,
                        span: initializer.span,
                        initializer_path: Some(initializer.path.clone()),
                        field: Some(assignment.name.clone()),
                        assignment_span: Some(assignment.span),
                        blocker_span: Some(absolute_span(
                            assignment.value.span.start,
                            unsupported.span,
                        )),
                        raw_value: Some(assignment.value.raw.clone()),
                    });
                }
            }
        }
        Ok(datum)
    }

    fn skip_initializer(
        &mut self,
        coordinate: WorldCoordinate,
        initializer: &PlannedInitializer,
        kind: WorldAllocationWorkKind,
    ) {
        self.stats.skipped_initializers += 1;
        self.work_items.push(WorldAllocationWorkItem {
            kind,
            coordinate,
            span: initializer.span,
            initializer_path: Some(initializer.path.clone()),
            field: None,
            assignment_span: None,
            blocker_span: None,
            raw_value: None,
        });
    }
}

fn absolute_span(base: usize, relative: SourceSpan) -> SourceSpan {
    SourceSpan::new(
        base.saturating_add(relative.start),
        base.saturating_add(relative.end),
    )
}

/// Fatal disagreement between a world plan and runtime image.
#[derive(Debug)]
pub enum WorldAllocationError {
    /// Runtime image allocation or heap mutation failed.
    Runtime(RuntimeImageError),
    /// A planned canonical path cannot be represented by runtime values.
    Value(ValueError),
}

impl fmt::Display for WorldAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => write!(formatter, "runtime allocation failed: {error}"),
            Self::Value(error) => write!(formatter, "invalid planned runtime value: {error}"),
        }
    }
}

impl std::error::Error for WorldAllocationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(error) => Some(error),
            Self::Value(error) => Some(error),
        }
    }
}

impl From<RuntimeImageError> for WorldAllocationError {
    fn from(error: RuntimeImageError) -> Self {
        Self::Runtime(error)
    }
}

impl From<ValueError> for WorldAllocationError {
    fn from(error: ValueError) -> Self {
        Self::Value(error)
    }
}
