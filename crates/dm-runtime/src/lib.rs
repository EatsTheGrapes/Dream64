//! Deterministic materialization of compiler constants into a runtime heap.
//!
//! This crate is intentionally a boundary between frontend identities and the
//! persistent runtime. Object-tree node IDs are consumed while building the
//! image but are never retained; canonical paths are the durable type keys.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dm_compiler::{Compilation, CompilerDatabase, CompilerError};
use dm_core::{FileId, SourceSpan};
use dm_globals::{
    ConstantEvaluation, ConstantListEntry, ConstantValue, InitializationStep, StorageClass,
    UnsupportedCategory, UnsupportedConstant, VariableEntry, VariableRegistry, evaluate_constant,
};
use dm_lexer::{TokenKind, lex};
use dm_object_tree::NodeKind;
use dm_semantics::ProcedureRegistry;
use dm_value::{DatumDefaults, DatumId, FieldName, TypePath, Value, ValueError, ValueHeap};
use dm_vm::{
    ExecutionContext, ExecutionState, InitializerBinding, InitializerProgram, Module, RuntimeError,
    compile_initializer, compile_initializer_into_module, execute_module_in_context,
};

/// A successfully materialized global or type-static variable.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeVariable {
    /// Canonical variable path from the object tree.
    pub path: String,
    /// Global or type-static storage lifetime.
    pub storage: StorageClass,
    /// Last constant value assigned in project source order.
    pub value: Value,
    /// Declaration ordinal of the assignment that produced `value`.
    pub ordinal: usize,
}

/// A runtime initializer deliberately left for a later execution phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeInitializerDiagnostic {
    /// Canonical variable path.
    pub variable_path: String,
    /// Storage lifetime that eventually receives the value.
    pub storage: StorageClass,
    /// Expanded declaration ordinal.
    pub ordinal: usize,
    /// Physical source file.
    pub file_id: FileId,
    /// Project-relative source path.
    pub source_path: String,
    /// Complete initializer span in original source bytes.
    pub initializer_span: SourceSpan,
    /// Precise unsupported token span in original source bytes.
    pub blocker_span: SourceSpan,
    /// Conservative reason materialization stopped.
    pub category: UnsupportedCategory,
    /// Runtime phase that rejected the initializer.
    pub phase: InitializerFailurePhase,
    /// Recoverable lowering or execution detail.
    pub message: String,
}

/// Phase that retained an initializer for a later compatibility pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitializerFailurePhase {
    /// Conservative constant evaluation rejected syntax not supported by the VM.
    ConstantEvaluation,
    /// VM expression lowering could not resolve or represent the expression.
    Lowering,
    /// Valid bytecode failed while reading current runtime state.
    Execution,
}

/// Result of conservatively applying one field expression to a live datum.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstantFieldApplication {
    /// The expression was proven constant and stored on the datum.
    Applied,
    /// Runtime evaluation or unsupported syntax is still required.
    Unsupported(UnsupportedConstant),
}

/// Canonical runtime metadata and direct defaults for one object type.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeType {
    path: TypePath,
    parent: Option<TypePath>,
    defaults: DatumDefaults,
}

fn materialize_builtin_atom_defaults(
    heap: &mut ValueHeap,
    datum: DatumId,
    is_atom: bool,
    is_movable: bool,
) -> Result<(), ValueError> {
    // Every /datum has BYOND's built-in tag field, even though it has no
    // source declaration in user projects.
    let tag = FieldName::parse("tag").expect("built-in datum field name is valid");
    if heap.datum_field(datum, &tag).is_err() {
        heap.set_datum_field(datum, tag, Value::Null)?;
    }
    if !is_atom {
        return Ok(());
    }
    // These names exist without source declarations in BYOND's built-in atom
    // hierarchy.  Materialize only absent values so a project declaration on a
    // descendant retains its normal inherited/default-layer precedence.
    let atom_defaults: &[(&str, Value)] = &[
        ("alpha", Value::number(255.0)),
        ("appearance_flags", Value::number(0.0)),
        ("blend_mode", Value::number(0.0)),
        ("color", Value::Null),
        ("density", Value::number(0.0)),
        ("dir", Value::number(2.0)),
        ("icon", Value::Null),
        ("icon_state", Value::Null),
        ("invisibility", Value::number(0.0)),
        ("layer", Value::number(1.0)),
        ("loc", Value::Null),
        ("maptext", Value::Null),
        ("maptext_height", Value::number(32.0)),
        ("maptext_width", Value::number(32.0)),
        ("mouse_opacity", Value::number(1.0)),
        ("name", Value::Null),
        ("opacity", Value::number(0.0)),
        ("overlays", Value::Null),
        ("plane", Value::number(0.0)),
        ("pixel_w", Value::number(0.0)),
        ("pixel_z", Value::number(0.0)),
        ("render_source", Value::Null),
        ("render_target", Value::Null),
        ("transform", Value::Null),
        ("underlays", Value::Null),
        ("vis_contents", Value::Null),
        ("vis_flags", Value::number(0.0)),
        ("x", Value::number(0.0)),
        ("y", Value::number(0.0)),
        ("z", Value::number(0.0)),
    ];
    let movable_defaults: &[(&str, Value)] = &[
        ("animate_movement", Value::number(0.0)),
        ("bound_height", Value::number(32.0)),
        ("bound_width", Value::number(32.0)),
        ("bound_x", Value::number(0.0)),
        ("bound_y", Value::number(0.0)),
        ("glide_size", Value::number(0.0)),
        ("pixel_x", Value::number(0.0)),
        ("pixel_y", Value::number(0.0)),
        ("screen_loc", Value::Null),
        ("step_size", Value::number(32.0)),
    ];
    for (name, value) in atom_defaults
        .iter()
        .chain(is_movable.then_some(movable_defaults).into_iter().flatten())
    {
        let name = FieldName::parse(name).expect("built-in atom field name is valid");
        if heap.datum_field(datum, &name).is_err() {
            heap.set_datum_field(datum, name, value.clone())?;
        }
    }
    Ok(())
}

fn materialize_builtin_world_defaults(
    heap: &mut ValueHeap,
    datum: DatumId,
    type_path: &TypePath,
    world_name: &str,
) -> Result<(), ValueError> {
    if type_path.as_str() != "/world" {
        return Ok(());
    }
    let system_type = if cfg!(windows) { "MS Windows" } else { "UNIX" };
    let defaults: &[(&str, Value)] = &[
        ("system_type", Value::text(system_type)),
        ("name", Value::text(world_name)),
        ("hub", Value::Null),
        ("hub_password", Value::Null),
        ("internet_address", Value::Null),
        ("address", Value::Null),
        ("status", Value::Null),
        ("port", Value::number(0.0)),
        (
            "area",
            Value::TypePath(TypePath::parse("/area").expect("built-in area path")),
        ),
        (
            "mob",
            Value::TypePath(TypePath::parse("/mob").expect("built-in mob path")),
        ),
        (
            "turf",
            Value::TypePath(TypePath::parse("/turf").expect("built-in turf path")),
        ),
        ("byond_version", Value::number(516.0)),
        ("byond_build", Value::number(1663.0)),
        ("cache_lifespan", Value::number(30.0)),
        ("executor", Value::Null),
        ("game_state", Value::number(0.0)),
        ("host", Value::Null),
        ("loop_checks", Value::number(1.0)),
        ("map_format", Value::number(0.0)),
        ("map_cpu", Value::number(0.0)),
        ("movement_mode", Value::number(0.0)),
        ("process", Value::number(std::process::id() as f32)),
        ("reachable", Value::number(0.0)),
        ("sleep_offline", Value::number(0.0)),
        ("tick_usage", Value::number(0.0)),
        ("url", Value::Null),
        ("version", Value::number(0.0)),
        ("view", Value::number(5.0)),
        ("visibility", Value::number(1.0)),
        ("icon_size", Value::number(32.0)),
        ("tick_lag", Value::number(1.0)),
        ("fps", Value::number(10.0)),
        ("timezone", Value::number(0.0)),
        ("cpu", Value::number(0.0)),
        ("time", Value::number(0.0)),
        ("timeofday", Value::number(0.0)),
        ("realtime", Value::number(0.0)),
    ];
    for (name, value) in defaults {
        let name = FieldName::parse(name).expect("built-in world field is valid");
        if heap.datum_field(datum, &name).is_err() {
            heap.set_datum_field(datum, name, value.clone())?;
        }
    }
    // DreamDaemon exposes an empty parameter list on an ordinary launch.
    // TGS and other portable libraries intentionally index it without first
    // checking for null; a host-provided `-params` value replaces this list.
    let params = FieldName::parse("params").expect("built-in world field is valid");
    if heap.datum_field(datum, &params).is_err() {
        let list = heap.allocate_list();
        heap.set_datum_field(datum, params, Value::List(list))?;
    }
    let log = FieldName::parse("log").expect("built-in world field is valid");
    if heap.datum_field(datum, &log).is_err() {
        heap.set_datum_field(datum, log, Value::Null)?;
    }
    let contents = FieldName::parse("contents").expect("built-in world field is valid");
    if heap.datum_field(datum, &contents).is_err() {
        let list = heap.allocate_list();
        heap.set_datum_field(datum, contents, Value::List(list))?;
    }
    Ok(())
}

fn builtin_initial_fields(path: &TypePath, world_name: &str) -> BTreeMap<FieldName, Value> {
    let mut fields = BTreeMap::new();
    let mut insert = |name: &str, value: Value| {
        fields.insert(
            FieldName::parse(name).expect("built-in initial field name is valid"),
            value,
        );
    };
    match path.as_str() {
        "/datum" => insert("tag", Value::Null),
        "/atom" => {
            for (name, value) in [
                ("alpha", Value::number(255.0)),
                ("appearance_flags", Value::number(0.0)),
                ("blend_mode", Value::number(0.0)),
                ("color", Value::Null),
                ("density", Value::number(0.0)),
                ("dir", Value::number(2.0)),
                ("icon", Value::Null),
                ("icon_state", Value::Null),
                ("invisibility", Value::number(0.0)),
                ("layer", Value::number(1.0)),
                ("loc", Value::Null),
                ("opacity", Value::number(0.0)),
                ("overlays", Value::Null),
                ("plane", Value::number(0.0)),
                ("underlays", Value::Null),
                ("vis_contents", Value::Null),
                ("x", Value::number(0.0)),
                ("y", Value::number(0.0)),
                ("z", Value::number(0.0)),
            ] {
                insert(name, value);
            }
        }
        "/atom/movable" => {
            for (name, value) in [
                ("animate_movement", Value::number(0.0)),
                ("bound_height", Value::number(32.0)),
                ("bound_width", Value::number(32.0)),
                ("bound_x", Value::number(0.0)),
                ("bound_y", Value::number(0.0)),
                ("glide_size", Value::number(0.0)),
                ("pixel_x", Value::number(0.0)),
                ("pixel_y", Value::number(0.0)),
                ("screen_loc", Value::Null),
                ("step_size", Value::number(32.0)),
            ] {
                insert(name, value);
            }
        }
        "/mob" => {
            insert("see_invisible", Value::number(0.0));
            insert("sight", Value::number(0.0));
        }
        "/image" => {
            for (name, value) in [
                ("alpha", Value::number(255.0)),
                ("appearance_flags", Value::number(0.0)),
                ("blend_mode", Value::number(0.0)),
                ("color", Value::Null),
                ("dir", Value::number(2.0)),
                ("icon", Value::Null),
                ("icon_state", Value::Null),
                ("layer", Value::number(0.0)),
                ("loc", Value::Null),
                ("name", Value::Null),
                ("overlays", Value::Null),
                ("plane", Value::number(0.0)),
                ("transform", Value::Null),
                ("underlays", Value::Null),
                ("vis_contents", Value::Null),
            ] {
                insert(name, value);
            }
        }
        "/world" => {
            insert(
                "system_type",
                Value::text(if cfg!(windows) { "MS Windows" } else { "UNIX" }),
            );
            insert("icon_size", Value::number(32.0));
            insert("tick_lag", Value::number(1.0));
            insert("fps", Value::number(10.0));
            insert("params", Value::Null);
            insert("name", Value::text(world_name));
            insert("hub", Value::Null);
            insert("hub_password", Value::Null);
            insert("internet_address", Value::Null);
            insert("address", Value::Null);
            insert("status", Value::Null);
            insert("port", Value::number(0.0));
            insert(
                "area",
                Value::TypePath(TypePath::parse("/area").expect("built-in area path")),
            );
            insert(
                "mob",
                Value::TypePath(TypePath::parse("/mob").expect("built-in mob path")),
            );
            insert(
                "turf",
                Value::TypePath(TypePath::parse("/turf").expect("built-in turf path")),
            );
            insert("byond_version", Value::number(516.0));
            insert("byond_build", Value::number(1663.0));
            insert("cache_lifespan", Value::number(30.0));
            insert("executor", Value::Null);
            insert("game_state", Value::number(0.0));
            insert("host", Value::Null);
            insert("loop_checks", Value::number(1.0));
            insert("map_format", Value::number(0.0));
            insert("map_cpu", Value::number(0.0));
            insert("movement_mode", Value::number(0.0));
            insert("process", Value::number(std::process::id() as f32));
            insert("reachable", Value::number(0.0));
            insert("sleep_offline", Value::number(0.0));
            insert("tick_usage", Value::number(0.0));
            insert("url", Value::Null);
            insert("version", Value::number(0.0));
            insert("view", Value::number(5.0));
            insert("visibility", Value::number(1.0));
        }
        _ => {}
    }
    fields
}

impl RuntimeType {
    /// Returns the canonical type path.
    #[must_use]
    pub const fn path(&self) -> &TypePath {
        &self.path
    }

    /// Returns the effective canonical parent type, when one exists.
    #[must_use]
    pub const fn parent(&self) -> Option<&TypePath> {
        self.parent.as_ref()
    }

    /// Returns defaults declared directly on this type.
    #[must_use]
    pub const fn defaults(&self) -> &DatumDefaults {
        &self.defaults
    }
}

/// Deterministic materialization counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeImageStats {
    /// Variable declarations inventoried by the frontend.
    pub variables: usize,
    /// Explicit initializer steps attempted.
    pub initializer_steps: usize,
    /// Constant steps successfully converted.
    pub constants_materialized: usize,
    /// Nonconstant initializer steps successfully executed by the VM.
    pub dynamic_initializers_materialized: usize,
    /// Unique global/static slots with a materialized value.
    pub runtime_variables: usize,
    /// Canonical types retained for later allocation.
    pub runtime_types: usize,
    /// Direct type-default layers containing at least one constant field.
    pub default_layers: usize,
    /// Constant list objects allocated, including nested lists.
    pub constant_lists: usize,
    /// Initializers retained for a future runtime phase.
    pub unsupported_initializers: usize,
    /// Datums allocated after image construction.
    pub datums_allocated: usize,
    /// Per-type instance-initializer plans compiled on first allocation.
    pub instance_initializer_plans_compiled: usize,
    /// Immutable type metadata snapshots built for execution-state transfers.
    pub execution_metadata_builds: usize,
    /// Per-type inherited-default allocation plans built on first allocation.
    pub datum_allocation_plans_built: usize,
    /// Datums allocated inside a caller-owned persistent execution state.
    pub stateful_datums_allocated: usize,
}

/// Result of compiling instance-initializer plans without allocating datums.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InitializerPreflightStats {
    /// Unique requested runtime types.
    pub types: usize,
    /// Plans compiled during this preflight call.
    pub plans_compiled: usize,
    /// Plans that were already cached before this call.
    pub plans_reused: usize,
}

/// A deterministic runtime-ready constant image for one compiled project.
pub struct RuntimeImage {
    heap: ValueHeap,
    variables: Vec<RuntimeVariable>,
    types: BTreeMap<TypePath, RuntimeType>,
    type_paths: Arc<BTreeSet<TypePath>>,
    type_parents: Arc<BTreeMap<TypePath, Option<TypePath>>>,
    initial_values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,
    world_name: String,
    canonical_world: Option<DatumId>,
    binding_index: RuntimeBindingIndex,
    global_variable_indices: BTreeMap<FieldName, usize>,
    diagnostics: Vec<RuntimeInitializerDiagnostic>,
    instance_initializers: Vec<(VariableEntry, InitializationStep)>,
    instance_initializer_plans: BTreeMap<TypePath, Arc<[CompiledInstanceInitializer]>>,
    datum_allocation_plans: BTreeMap<TypePath, DatumAllocationPlan>,
    project_root: PathBuf,
    stats: RuntimeImageStats,
}

#[derive(Clone)]
struct CompiledInstanceInitializer {
    path: String,
    field: FieldName,
    program: Arc<InitializerProgram>,
}

#[derive(Clone)]
struct DatumAllocationPlan {
    defaults: Arc<[DatumDefaults]>,
    ancestors: Arc<BTreeSet<TypePath>>,
    is_atom: bool,
    is_movable: bool,
}

struct DynamicInitializerFailure {
    phase: InitializerFailurePhase,
    message: String,
    expanded_span: SourceSpan,
}

struct RuntimeBindingIndex {
    globals: BTreeMap<String, FieldName>,
    statics: BTreeMap<String, FieldName>,
    instance_fields: BTreeMap<String, BTreeMap<String, FieldName>>,
}

impl RuntimeBindingIndex {
    fn build(registry: &VariableRegistry) -> Result<Self, RuntimeImageError> {
        let mut globals = BTreeMap::new();
        let mut statics = BTreeMap::new();
        let mut instance_fields = BTreeMap::<String, BTreeMap<String, FieldName>>::new();
        for entry in registry.entries() {
            let field = variable_field(&entry.path)?;
            if entry.storage == StorageClass::Instance {
                if let Some(owner) = &entry.owner {
                    instance_fields
                        .entry(owner.path.clone())
                        .or_default()
                        .insert(field.as_str().to_owned(), field);
                }
            } else if entry.storage == StorageClass::Global {
                globals.insert(field.as_str().to_owned(), field);
            } else {
                statics.insert(entry.path.clone(), FieldName::static_storage(&entry.path));
            }
        }
        Ok(Self {
            globals,
            statics,
            instance_fields,
        })
    }
}

fn execution_metadata(
    types: &BTreeMap<TypePath, RuntimeType>,
    world_name: &str,
) -> (
    BTreeMap<TypePath, Option<TypePath>>,
    BTreeMap<TypePath, BTreeMap<FieldName, Value>>,
) {
    let type_parents = types
        .iter()
        .map(|(path, runtime_type)| (path.clone(), runtime_type.parent.clone()))
        .collect();
    let mut initial_values = BTreeMap::new();
    for path in types.keys() {
        let mut hierarchy = Vec::new();
        let mut current = Some(path.clone());
        let mut visited = BTreeSet::new();
        while let Some(candidate) = current.take() {
            if !visited.insert(candidate.clone()) {
                break;
            }
            let Some(runtime_type) = types.get(&candidate) else {
                break;
            };
            hierarchy.push(candidate.clone());
            current.clone_from(&runtime_type.parent);
        }
        hierarchy.reverse();
        let mut values = BTreeMap::new();
        for ancestor in hierarchy {
            values.extend(builtin_initial_fields(&ancestor, world_name));
            if let Some(runtime_type) = types.get(&ancestor) {
                values.extend(
                    runtime_type
                        .defaults
                        .fields()
                        .map(|(field, value)| (field.clone(), value.clone())),
                );
            }
        }
        values.insert(
            FieldName::parse("type").expect("built-in type field is valid"),
            Value::TypePath(path.clone()),
        );
        values.insert(
            FieldName::parse("parent_type").expect("built-in parent_type field is valid"),
            types
                .get(path)
                .and_then(|runtime_type| runtime_type.parent.clone())
                .map_or(Value::Null, Value::TypePath),
        );
        initial_values.insert(path.clone(), values);
    }
    (type_parents, initial_values)
}

impl RuntimeImage {
    fn refresh_execution_metadata(&mut self) {
        let (type_parents, initial_values) = execution_metadata(&self.types, &self.world_name);
        self.type_parents = Arc::new(type_parents);
        self.initial_values = Arc::new(initial_values);
        self.stats.execution_metadata_builds += 1;
    }
    /// Compiles and materializes a project without allocating map atoms.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageLoadError`] for project loading or an invalid
    /// canonical path produced at the frontend/runtime boundary.
    pub fn load(root_file: impl AsRef<Path>) -> Result<Self, RuntimeImageLoadError> {
        let compilation = CompilerDatabase::new()
            .compile(root_file)
            .map_err(RuntimeImageLoadError::Compiler)?;
        Self::from_compilation(&compilation).map_err(RuntimeImageLoadError::Image)
    }

    /// Materializes one existing frontend snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageError`] if a frontend path cannot be represented
    /// by the runtime's canonical path or field-name types.
    pub fn from_compilation(compilation: &Compilation) -> Result<Self, RuntimeImageError> {
        let registry = VariableRegistry::build(compilation);
        let plans = registry.initialization_plans();
        let binding_index = RuntimeBindingIndex::build(&registry)?;
        let mut types = runtime_types(compilation)?;
        // BYOND materializes every declared instance variable on every datum,
        // even when the declaration has no explicit initializer. Seed those
        // fields with null before applying constant/dynamic default layers;
        // otherwise a valid bare read is misdiagnosed as a missing field.
        for entry in registry
            .entries()
            .iter()
            .filter(|entry| entry.storage == StorageClass::Instance)
        {
            let Some(owner) = &entry.owner else {
                continue;
            };
            let owner_path = parse_type_path(&owner.path)?;
            let field = variable_field(&entry.path)?;
            if let Some(runtime_type) = types.get_mut(&owner_path) {
                runtime_type.defaults.set(field, Value::Null);
            }
        }
        let type_paths = Arc::new(types.keys().cloned().collect());
        let world_name = compilation
            .project()
            .files
            .first()
            .and_then(|file| file.relative_path.file_stem())
            .and_then(|stem| stem.to_str())
            .unwrap_or("world")
            .to_owned();
        let (type_parents, initial_values) = execution_metadata(&types, &world_name);
        let mut image = Self {
            heap: ValueHeap::new(),
            variables: Vec::new(),
            types,
            type_paths,
            type_parents: Arc::new(type_parents),
            initial_values: Arc::new(initial_values),
            world_name,
            canonical_world: None,
            binding_index,
            global_variable_indices: BTreeMap::new(),
            diagnostics: Vec::new(),
            instance_initializers: Vec::new(),
            instance_initializer_plans: BTreeMap::new(),
            datum_allocation_plans: BTreeMap::new(),
            project_root: compilation.project().root_directory.clone(),
            stats: RuntimeImageStats {
                variables: registry.entries().len(),
                execution_metadata_builds: 1,
                initializer_steps: plans.global_steps.len()
                    + plans
                        .type_defaults
                        .iter()
                        .map(|plan| plan.steps.len())
                        .sum::<usize>(),
                ..RuntimeImageStats::default()
            },
        };

        // An initializer plan contains only declarations with an explicit
        // `=` expression. BYOND still installs every plain instance variable
        // as a real null-valued field in the object tree. Seed those fields
        // before any global initializer can execute `new /type`; otherwise a
        // datum created by a global initializer has an incomplete shape and a
        // legitimate read/compound assignment reports "missing field".
        for entry in registry.entries().iter().filter(|entry| {
            entry.storage == StorageClass::Instance
                && entry.assignment == dm_globals::AssignmentKind::Declaration
                && entry.initializer.is_none()
        }) {
            let owner = entry
                .owner
                .as_ref()
                .ok_or_else(|| RuntimeImageError::MissingOwner(entry.path.clone()))?;
            let owner = parse_type_path(&owner.path)?;
            let field = variable_field(&entry.path)?;
            image
                .types
                .get_mut(&owner)
                .ok_or_else(|| RuntimeImageError::UnknownType(owner.clone()))?
                .defaults
                .set(field, Value::Null);
        }
        // Object-tree defaults are a compile-time phase in BYOND. A global
        // initializer may appear textually before the type body it constructs,
        // but `new` must still observe the type's complete defaults. Materialize
        // scalar constants now and retain list/dynamic expressions for fresh
        // per-instance evaluation before executing any global/static code.
        let mut instance_steps = plans
            .type_defaults
            .iter()
            .flat_map(|plan| plan.steps.iter())
            .collect::<Vec<_>>();
        instance_steps.sort_by_key(|step| step.ordinal);
        image.refresh_execution_metadata();
        let mut state = image.take_execution_state();
        for step in instance_steps {
            let entry = &registry.entries()[step.entry_index];
            match &step.evaluation {
                ConstantEvaluation::Value(ConstantValue::List(_))
                | ConstantEvaluation::Unsupported(_) => image
                    .instance_initializers
                    .push((entry.clone(), step.clone())),
                ConstantEvaluation::Value(constant) => {
                    let value = image.convert_constant_in(constant, state.heap_mut())?;
                    image.apply_step_value(entry, step, value)?;
                    image.stats.constants_materialized += 1;
                }
            }
        }
        image.restore_execution_state(state);
        image.refresh_execution_metadata();

        // BYOND's canonical world singleton exists before global/static
        // initialization and remains the same object through Genesis. Its
        // constant layers are now complete because the object-tree phase above
        // ran first. Keep this bootstrap allocation outside ordinary allocation
        // counters and lazy per-type plan caches.
        let world_path = TypePath::parse("/world").expect("canonical world path is valid");
        if image.types.contains_key(&world_path) {
            let layers = image.default_layers(&world_path).map_err(|failure| {
                RuntimeImageError::InstanceInitializer {
                    path: "/world".to_owned(),
                    message: failure.message,
                }
            })?;
            let world = image
                .heap
                .allocate_datum_with_defaults(world_path.clone(), &layers);
            materialize_builtin_world_defaults(
                &mut image.heap,
                world,
                &world_path,
                &image.world_name,
            )?;
            image.canonical_world = Some(world);
        }

        // Global/static initializers form one ordered execution phase. Keep their
        // globals and heap in a single state: constructing a new state for
        // every initializer copies the entire growing heap (and loses writes
        // made by an earlier initializer).
        let mut state = image.take_execution_state();
        // Build the project procedure module once. Initializer entry points are
        // appended to this shared module, avoiding an O(module) clone per
        // declaration while retaining ordinary and dynamic call targets.
        let mut initializer_module = ProcedureRegistry::build(compilation)
            .compile_vm(compilation)
            .ok()
            .map(|executable| executable.module().clone());
        let mut global_steps = plans.global_steps.iter().collect::<Vec<_>>();
        global_steps.sort_by_key(|step| step.ordinal);
        for step in global_steps {
            let entry = &registry.entries()[step.entry_index];
            match &step.evaluation {
                ConstantEvaluation::Value(constant) => {
                    let value = image.convert_constant_in(constant, state.heap_mut())?;
                    image.apply_step_value(entry, step, value)?;
                    image.sync_initializer_global(step, &mut state)?;
                    image.stats.constants_materialized += 1;
                }
                ConstantEvaluation::Unsupported(unsupported) => {
                    match image.execute_dynamic_initializer(
                        entry,
                        step,
                        &mut state,
                        initializer_module.as_mut(),
                    ) {
                        Ok(value) => {
                            image.apply_step_value(entry, step, value)?;
                            image.sync_initializer_global(step, &mut state)?;
                            image.stats.dynamic_initializers_materialized += 1;
                        }
                        Err(failure) => image.retain_dynamic_failure(
                            compilation,
                            entry,
                            step,
                            unsupported,
                            failure,
                        )?,
                    }
                }
            }
        }
        image.restore_execution_state(state);
        image.refresh_execution_metadata();
        image.stats.runtime_variables = image.variables.len();
        image.stats.runtime_types = image.types.len();
        image.stats.default_layers = image
            .types
            .values()
            .filter(|runtime_type| runtime_type.defaults.fields().len() != 0)
            .count();
        image.stats.unsupported_initializers = image.diagnostics.len();
        Ok(image)
    }

    /// Returns the runtime value heap.
    #[must_use]
    pub const fn heap(&self) -> &ValueHeap {
        &self.heap
    }

    /// Returns the runtime value heap for later execution integration.
    #[must_use]
    pub const fn heap_mut(&mut self) -> &mut ValueHeap {
        &mut self.heap
    }

    /// Returns materialized global/static slots in first-encounter order.
    #[must_use]
    pub fn variables(&self) -> &[RuntimeVariable] {
        &self.variables
    }

    /// Returns the canonical `/world` singleton allocated before globals.
    #[must_use]
    pub const fn canonical_world(&self) -> Option<DatumId> {
        self.canonical_world
    }

    /// Looks up a materialized global/static slot by canonical variable path.
    #[must_use]
    pub fn variable(&self, path: &str) -> Option<&RuntimeVariable> {
        self.variables.iter().find(|variable| variable.path == path)
    }

    /// Iterates canonical types in lexical path order.
    pub fn types(&self) -> impl Iterator<Item = (&TypePath, &RuntimeType)> {
        self.types.iter()
    }

    /// Returns retained unsupported initializers in project source order.
    #[must_use]
    pub fn diagnostics(&self) -> &[RuntimeInitializerDiagnostic] {
        &self.diagnostics
    }

    /// Returns deterministic materialization counters.
    #[must_use]
    pub const fn stats(&self) -> &RuntimeImageStats {
        &self.stats
    }

    /// Compiles and caches instance-initializer plans for a set of runtime types.
    ///
    /// No datums are allocated and no initializer bytecode is executed. Input is
    /// deduplicated and processed in canonical path order, so aggregated failures
    /// are deterministic regardless of iterator order.
    ///
    /// # Errors
    ///
    /// Returns every invalid type, inheritance, binding, or lowering failure in
    /// canonical type order. Successfully prepared plans remain cached.
    pub fn preflight_instance_initializers(
        &mut self,
        type_paths: impl IntoIterator<Item = TypePath>,
    ) -> Result<InitializerPreflightStats, Vec<RuntimeImageError>> {
        let paths = type_paths.into_iter().collect::<BTreeSet<_>>();
        let mut stats = InitializerPreflightStats {
            types: paths.len(),
            ..InitializerPreflightStats::default()
        };
        let mut errors = Vec::new();
        for path in paths {
            let reused = self.instance_initializer_plans.contains_key(&path);
            let allocation = match self.datum_allocation_plan(&path) {
                Ok(plan) => plan,
                Err(error) => {
                    errors.push(error);
                    continue;
                }
            };
            match self.instance_initializer_plan(&path, &allocation.ancestors) {
                Ok(_) if reused => stats.plans_reused += 1,
                Ok(_) => stats.plans_compiled += 1,
                Err(error) => errors.push(error),
            }
        }
        if errors.is_empty() {
            Ok(stats)
        } else {
            Err(errors)
        }
    }

    /// Transfers the shared heap and materialized DM globals into VM state.
    ///
    /// # Panics
    ///
    /// Panics only if an engine-defined built-in field name is internally invalid;
    /// every such spelling is a fixed canonical DM identifier.
    #[must_use]
    pub fn take_execution_state(&mut self) -> ExecutionState {
        let mut state = ExecutionState::from_heap(std::mem::take(&mut self.heap));
        state.set_shared_type_paths(Arc::clone(&self.type_paths));
        state.set_shared_type_parents(Arc::clone(&self.type_parents));
        state.set_shared_initial_values(Arc::clone(&self.initial_values));
        state.set_project_root(self.project_root.clone());
        if let Some(world) = self.canonical_world {
            state.set_global(
                FieldName::parse("world").expect("built-in world global name is valid"),
                Value::Datum(world),
            );
        }
        for field in self.binding_index.globals.values() {
            state.set_global(field.clone(), Value::Null);
        }
        for field in self.binding_index.statics.values() {
            state.set_global(field.clone(), Value::Null);
        }
        for (field, index) in &self.global_variable_indices {
            let value = self.variables[*index].value.clone();
            state.set_global(field.clone(), value.clone());
            state.set_initial_global(field.clone(), value);
        }
        for variable in self
            .variables
            .iter()
            .filter(|variable| variable.storage == StorageClass::Static)
        {
            state.set_global(
                FieldName::static_storage(&variable.path),
                variable.value.clone(),
            );
        }
        state
    }

    /// Restores VM heap state and captures mutations to materialized globals.
    ///
    /// Values written to globals unknown to this image are retained only by the
    /// supplied state; declared globals are synchronized by their DM field name.
    pub fn restore_execution_state(&mut self, state: ExecutionState) {
        for (field, index) in &self.global_variable_indices {
            if let Some(value) = state.global(field) {
                self.variables[*index].value.clone_from(value);
            }
        }
        self.heap = state.into_heap();
    }

    /// Allocates one datum with all constant ancestor defaults applied.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageError::UnknownType`] for absent metadata or
    /// [`RuntimeImageError::InheritanceCycle`] for an invalid retained chain.
    pub fn allocate_datum(&mut self, type_path: &TypePath) -> Result<DatumId, RuntimeImageError> {
        let allocation = self.datum_allocation_plan(type_path)?;
        let datum = self
            .heap
            .allocate_datum_with_defaults(type_path.clone(), &allocation.defaults);
        materialize_builtin_atom_defaults(
            &mut self.heap,
            datum,
            allocation.is_atom,
            allocation.is_movable,
        )?;
        materialize_builtin_world_defaults(&mut self.heap, datum, type_path, &self.world_name)?;
        let plan = self.instance_initializer_plan(type_path, &allocation.ancestors)?;
        if !plan.is_empty() {
            let mut state = self.take_execution_state();
            let result: Result<(), RuntimeImageError> = (|| {
                for initializer in plan.iter() {
                    let value = execute_module_in_context(
                        initializer.program.module(),
                        initializer.program.entry(),
                        &[],
                        &mut state,
                        &ExecutionContext::new(Value::Datum(datum), Value::Null),
                    )
                    .map_err(|error| {
                        RuntimeImageError::InstanceInitializer {
                            path: initializer.path.clone(),
                            message: error.message,
                        }
                    })?;
                    state
                        .heap_mut()
                        .set_datum_field(datum, initializer.field.clone(), value)?;
                }
                Ok(())
            })();
            self.restore_execution_state(state);
            result?;
            self.stats.dynamic_initializers_materialized += plan.len();
        }
        self.stats.datums_allocated += 1;
        Ok(datum)
    }

    /// Allocates a datum directly in an existing execution state.
    ///
    /// This preserves the same defaults and initializer semantics as
    /// [`Self::allocate_datum`] while allowing bulk allocators to reuse one VM
    /// global/heap state instead of transferring it for every datum.
    ///
    /// # Errors
    ///
    /// Returns the same type, inheritance, lowering, execution, and heap errors
    /// as [`Self::allocate_datum`].
    pub fn allocate_datum_in_state(
        &mut self,
        type_path: &TypePath,
        state: &mut ExecutionState,
    ) -> Result<DatumId, RuntimeImageError> {
        let allocation = self.datum_allocation_plan(type_path)?;
        let datum = state
            .heap_mut()
            .allocate_datum_with_defaults(type_path.clone(), &allocation.defaults);
        materialize_builtin_atom_defaults(
            state.heap_mut(),
            datum,
            allocation.is_atom,
            allocation.is_movable,
        )?;
        materialize_builtin_world_defaults(state.heap_mut(), datum, type_path, &self.world_name)?;
        let plan = self.instance_initializer_plan(type_path, &allocation.ancestors)?;
        for initializer in plan.iter() {
            let value = execute_module_in_context(
                initializer.program.module(),
                initializer.program.entry(),
                &[],
                state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            )
            .map_err(|error| RuntimeImageError::InstanceInitializer {
                path: initializer.path.clone(),
                message: error.message,
            })?;
            state
                .heap_mut()
                .set_datum_field(datum, initializer.field.clone(), value)?;
        }
        self.stats.dynamic_initializers_materialized += plan.len();
        self.stats.datums_allocated += 1;
        self.stats.stateful_datums_allocated += 1;
        Ok(datum)
    }

    fn datum_allocation_plan(
        &mut self,
        type_path: &TypePath,
    ) -> Result<DatumAllocationPlan, RuntimeImageError> {
        if let Some(plan) = self.datum_allocation_plans.get(type_path) {
            return Ok(plan.clone());
        }
        let mut chain = Vec::new();
        let mut current = Some(type_path.clone());
        let mut visited = BTreeSet::new();
        let mut is_atom = false;
        let mut is_movable = false;
        while let Some(path) = current.take() {
            if !visited.insert(path.clone()) {
                return Err(RuntimeImageError::InheritanceCycle(path));
            }
            let runtime_type = self
                .types
                .get(&path)
                .ok_or_else(|| RuntimeImageError::UnknownType(path.clone()))?;
            is_atom |= path.as_str() == "/atom";
            is_movable |= path.as_str() == "/atom/movable";
            chain.push(runtime_type.defaults.clone());
            current.clone_from(&runtime_type.parent);
        }
        chain.reverse();
        let plan = DatumAllocationPlan {
            defaults: Arc::from(chain),
            ancestors: Arc::new(visited),
            is_atom,
            is_movable,
        };
        self.datum_allocation_plans
            .insert(type_path.clone(), plan.clone());
        self.stats.datum_allocation_plans_built += 1;
        Ok(plan)
    }

    fn instance_initializer_plan(
        &mut self,
        type_path: &TypePath,
        ancestors: &BTreeSet<TypePath>,
    ) -> Result<Arc<[CompiledInstanceInitializer]>, RuntimeImageError> {
        if let Some(plan) = self.instance_initializer_plans.get(type_path) {
            return Ok(plan.clone());
        }

        let applicable = self
            .instance_initializers
            .iter()
            .filter(|(entry, _)| {
                entry.owner.as_ref().is_some_and(|owner| {
                    TypePath::parse(&owner.path).is_ok_and(|owner| ancestors.contains(&owner))
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut plan = Vec::with_capacity(applicable.len());
        for (entry, step) in applicable {
            let initializer = entry
                .initializer
                .as_ref()
                .ok_or_else(|| RuntimeImageError::MissingInitializer(step.path.clone()))?;
            let bindings = self.initializer_bindings(&entry).map_err(|failure| {
                RuntimeImageError::InstanceInitializer {
                    path: step.path.clone(),
                    message: failure.message,
                }
            })?;
            let program =
                compile_initializer(&initializer.tokens, &bindings, None).map_err(|error| {
                    RuntimeImageError::InstanceInitializer {
                        path: step.path.clone(),
                        message: error.message,
                    }
                })?;
            plan.push(CompiledInstanceInitializer {
                path: step.path.clone(),
                field: variable_field(&step.path)?,
                program: Arc::new(program),
            });
        }
        let plan = Arc::<[CompiledInstanceInitializer]>::from(plan);
        self.instance_initializer_plans
            .insert(type_path.clone(), Arc::clone(&plan));
        self.stats.instance_initializer_plans_compiled += 1;
        Ok(plan)
    }

    /// Conservatively evaluates and applies one expression to a live datum field.
    ///
    /// This method executes no DM procedures. Unsupported expressions are
    /// returned to the caller with expression-relative source spans and leave
    /// the existing field value unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageError`] when a proven constant cannot be converted
    /// to a runtime value or `datum` is stale.
    pub fn apply_constant_field_expression(
        &mut self,
        datum: DatumId,
        field: FieldName,
        expression: &str,
    ) -> Result<ConstantFieldApplication, RuntimeImageError> {
        let tokens = match lex(expression) {
            Ok(tokens) => tokens
                .into_iter()
                .filter(|token| {
                    !matches!(
                        token.kind,
                        TokenKind::LineStart { .. }
                            | TokenKind::Newline
                            | TokenKind::LineContinuation
                    )
                })
                .collect::<Vec<_>>(),
            Err(error) => {
                return Ok(ConstantFieldApplication::Unsupported(UnsupportedConstant {
                    category: UnsupportedCategory::InvalidSyntax,
                    span: error.span,
                }));
            }
        };
        match evaluate_constant(&tokens) {
            ConstantEvaluation::Value(constant) => {
                let mut heap = std::mem::take(&mut self.heap);
                let result = self.convert_constant_in(&constant, &mut heap);
                self.heap = heap;
                let value = result?;
                self.heap.set_datum_field(datum, field, value)?;
                Ok(ConstantFieldApplication::Applied)
            }
            ConstantEvaluation::Unsupported(unsupported) => {
                Ok(ConstantFieldApplication::Unsupported(unsupported))
            }
        }
    }

    /// Applies one proven-constant field expression in a caller-owned execution state.
    ///
    /// # Errors
    ///
    /// Returns the same conversion and heap errors as
    /// [`Self::apply_constant_field_expression`].
    pub fn apply_constant_field_expression_in_state(
        &mut self,
        state: &mut ExecutionState,
        datum: DatumId,
        field: FieldName,
        expression: &str,
    ) -> Result<ConstantFieldApplication, RuntimeImageError> {
        let tokens = match lex(expression) {
            Ok(tokens) => tokens
                .into_iter()
                .filter(|token| {
                    !matches!(
                        token.kind,
                        TokenKind::LineStart { .. }
                            | TokenKind::Newline
                            | TokenKind::LineContinuation
                    )
                })
                .collect::<Vec<_>>(),
            Err(error) => {
                return Ok(ConstantFieldApplication::Unsupported(UnsupportedConstant {
                    category: UnsupportedCategory::InvalidSyntax,
                    span: error.span,
                }));
            }
        };
        match evaluate_constant(&tokens) {
            ConstantEvaluation::Value(constant) => {
                let value = self.convert_constant_in(&constant, state.heap_mut())?;
                state.heap_mut().set_datum_field(datum, field, value)?;
                Ok(ConstantFieldApplication::Applied)
            }
            ConstantEvaluation::Unsupported(unsupported) => {
                Ok(ConstantFieldApplication::Unsupported(unsupported))
            }
        }
    }

    /// Evaluates one raw expression against a live datum in an execution state.
    ///
    /// Bare identifiers resolve as fields inherited by `datum`'s runtime type,
    /// or as declared global variables. The supplied state owns both the live
    /// datum and the materialized global values, so writes performed by a later
    /// expression or lifecycle procedure remain visible to this evaluation.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageError::ExpressionLowering`] when the expression
    /// cannot be represented by the initializer VM subset, and
    /// [`RuntimeImageError::ExpressionExecution`] with the VM's source-mapped
    /// call stack when evaluation fails. It also returns [`RuntimeImageError`]
    /// when `datum` is stale or has an unknown runtime type.
    pub fn evaluate_datum_expression(
        &self,
        datum: DatumId,
        expression: &str,
        state: &mut ExecutionState,
    ) -> Result<Value, RuntimeImageError> {
        let tokens = lex(expression).map_err(|error| RuntimeImageError::ExpressionLowering {
            message: error.message,
            source_span: error.span,
        })?;
        let tokens = tokens
            .into_iter()
            .filter(|token| {
                !matches!(
                    token.kind,
                    TokenKind::LineStart { .. } | TokenKind::Newline | TokenKind::LineContinuation
                )
            })
            .collect::<Vec<_>>();
        let type_path = state.heap().datum(datum)?.type_path().clone();
        let bindings = self.datum_expression_bindings(&type_path)?;
        let program = compile_initializer(&tokens, &bindings, None).map_err(|error| {
            let source_span = tokens
                .first()
                .zip(tokens.last())
                .map_or(SourceSpan::new(0, 0), |(first, last)| {
                    SourceSpan::new(first.span.start, last.span.end)
                });
            RuntimeImageError::ExpressionLowering {
                message: error.message,
                source_span,
            }
        })?;
        execute_module_in_context(
            program.module(),
            program.entry(),
            &[],
            state,
            &ExecutionContext::new(Value::Datum(datum), Value::Null),
        )
        .map_err(RuntimeImageError::ExpressionExecution)
    }

    fn convert_constant_in(
        &mut self,
        constant: &ConstantValue,
        heap: &mut ValueHeap,
    ) -> Result<Value, RuntimeImageError> {
        Ok(match constant {
            ConstantValue::Null => Value::Null,
            ConstantValue::Number(number) => Value::Number(*number),
            ConstantValue::Text(text) => Value::text(text.as_str()),
            ConstantValue::TypePath(path) => Value::TypePath(parse_type_path(path)?),
            ConstantValue::List(entries) => {
                let list = heap.allocate_list();
                self.stats.constant_lists += 1;
                for entry in entries {
                    match entry {
                        ConstantListEntry::Positional(constant) => {
                            let value = self.convert_constant_in(constant, heap)?;
                            heap.list_mut(list)?.add(value);
                        }
                        ConstantListEntry::Associative { key, value } => {
                            let key = self.convert_constant_in(key, heap)?;
                            let value = self.convert_constant_in(value, heap)?;
                            heap.list_mut(list)?.set_key(key, value);
                        }
                    }
                }
                Value::List(list)
            }
        })
    }

    fn apply_step_value(
        &mut self,
        entry: &VariableEntry,
        step: &InitializationStep,
        value: Value,
    ) -> Result<(), RuntimeImageError> {
        if step.storage == StorageClass::Instance {
            let owner = entry
                .owner
                .as_ref()
                .ok_or_else(|| RuntimeImageError::MissingOwner(step.path.clone()))?;
            let owner = parse_type_path(&owner.path)?;
            let field = variable_field(&step.path)?;
            let runtime_type = self
                .types
                .get_mut(&owner)
                .ok_or_else(|| RuntimeImageError::UnknownType(owner.clone()))?;
            runtime_type.defaults.set(field, value);
            return Ok(());
        }
        if let Some(variable) = self
            .variables
            .iter_mut()
            .find(|variable| variable.path == step.path)
        {
            variable.value = value;
            variable.ordinal = step.ordinal;
            variable.storage = step.storage;
            return Ok(());
        }
        self.variables.push(RuntimeVariable {
            path: step.path.clone(),
            storage: step.storage,
            value,
            ordinal: step.ordinal,
        });
        if step.storage != StorageClass::Instance {
            let field = if step.storage == StorageClass::Global {
                variable_field(&step.path)?
            } else {
                FieldName::static_storage(&step.path)
            };
            self.global_variable_indices
                .insert(field, self.variables.len() - 1);
        }
        Ok(())
    }

    fn sync_initializer_global(
        &self,
        step: &InitializationStep,
        state: &mut ExecutionState,
    ) -> Result<(), RuntimeImageError> {
        if step.storage == StorageClass::Instance {
            return Ok(());
        }
        let field = if step.storage == StorageClass::Global {
            variable_field(&step.path)?
        } else {
            FieldName::static_storage(&step.path)
        };
        let value = self
            .variables
            .iter()
            .find(|variable| variable.path == step.path)
            .map(|variable| variable.value.clone())
            .ok_or_else(|| RuntimeImageError::MissingInitializer(step.path.clone()))?;
        state.set_global(field, value);
        Ok(())
    }

    fn execute_dynamic_initializer(
        &self,
        entry: &VariableEntry,
        step: &InitializationStep,
        state: &mut ExecutionState,
        linked_module: Option<&mut Module>,
    ) -> Result<Value, DynamicInitializerFailure> {
        let initializer = entry
            .initializer
            .as_ref()
            .ok_or_else(|| DynamicInitializerFailure {
                phase: InitializerFailurePhase::Lowering,
                message: format!("initialization step for {:?} has no syntax", step.path),
                expanded_span: entry.span,
            })?;
        let initializer_span = initializer.expanded_span;
        let bindings = self.initializer_bindings(entry)?;
        let standalone;
        let (module, entry_point) = if let Some(module) = linked_module {
            let entry = compile_initializer_into_module(&initializer.tokens, &bindings, module)
                .map_err(|error| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Lowering,
                    message: error.message,
                    expanded_span: initializer_span,
                })?;
            (&*module, entry)
        } else {
            standalone =
                compile_initializer(&initializer.tokens, &bindings, None).map_err(|error| {
                    DynamicInitializerFailure {
                        phase: InitializerFailurePhase::Lowering,
                        message: error.message,
                        expanded_span: initializer_span,
                    }
                })?;
            (standalone.module(), standalone.entry())
        };

        let src = if step.storage == StorageClass::Instance {
            let owner = entry
                .owner
                .as_ref()
                .ok_or_else(|| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Lowering,
                    message: format!("instance variable {:?} has no owning type", step.path),
                    expanded_span: initializer_span,
                })?;
            let owner =
                TypePath::parse(&owner.path).map_err(|error| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Lowering,
                    message: error.to_string(),
                    expanded_span: initializer_span,
                })?;
            let layers = self.default_layers(&owner).map_err(|mut failure| {
                failure.expanded_span = initializer_span;
                failure
            })?;
            Some(
                state
                    .heap_mut()
                    .allocate_datum_with_defaults(owner, layers.as_slice()),
            )
        } else {
            None
        };

        let context = ExecutionContext::new(src.map_or(Value::Null, Value::Datum), Value::Null);
        let result = execute_module_in_context(module, entry_point, &[], state, &context);
        if let Some(src) = src {
            let _ = state.heap_mut().destroy_datum(src);
        }
        match result {
            Ok(Value::Datum(_)) if step.storage == StorageClass::Instance => {
                Err(DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Execution,
                    message: "datum references require per-instance initialization".to_owned(),
                    expanded_span: initializer_span,
                })
            }
            Ok(value) => Ok(value),
            Err(error) => Err(DynamicInitializerFailure {
                phase: InitializerFailurePhase::Execution,
                message: error.message,
                expanded_span: error.source_span.unwrap_or(initializer_span),
            }),
        }
    }

    fn initializer_bindings(
        &self,
        entry: &VariableEntry,
    ) -> Result<BTreeMap<String, InitializerBinding>, DynamicInitializerFailure> {
        let mut bindings = self
            .binding_index
            .globals
            .iter()
            .map(|(name, field)| (name.clone(), InitializerBinding::Global(field.clone())))
            .collect::<BTreeMap<_, _>>();
        // `type` and `parent_type` are implicit datum variables in DM. They
        // participate in initializer expressions just like declared fields;
        // notably, TG code uses `parent_type::field` to inherit a constant
        // value without constructing an instance of the parent.
        for builtin in ["type", "parent_type"] {
            let field = FieldName::parse(builtin).expect("built-in datum field is valid");
            bindings.insert(builtin.to_owned(), InitializerBinding::SrcField(field));
        }
        if let Some(owner) = &entry.owner {
            let mut owners = Vec::new();
            let mut current = TypePath::parse(&owner.path).ok();
            while let Some(path) = current.take() {
                owners.push(path.clone());
                current = self.types.get(&path).and_then(|ty| ty.parent.clone());
            }
            owners.reverse();
            for owner in owners {
                let marker = format!("{}/var/", owner.as_str());
                for (path, storage) in &self.binding_index.statics {
                    if let Some(name) = path.strip_prefix(&marker)
                        && !name.contains('/')
                    {
                        bindings
                            .insert(name.to_owned(), InitializerBinding::Global(storage.clone()));
                    }
                }
            }
        }
        if entry.storage != StorageClass::Instance {
            return Ok(bindings);
        }

        let Some(owner) = &entry.owner else {
            return Ok(bindings);
        };
        let mut owners = Vec::new();
        let mut current =
            Some(
                TypePath::parse(&owner.path).map_err(|error| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Lowering,
                    message: error.to_string(),
                    expanded_span: entry.span,
                })?,
            );
        while let Some(path) = current.take() {
            owners.push(path.clone());
            current = self
                .types
                .get(&path)
                .and_then(|runtime_type| runtime_type.parent.clone());
        }
        owners.reverse();
        for owner in owners {
            if let Some(fields) = self.binding_index.instance_fields.get(owner.as_str()) {
                for (name, field) in fields {
                    bindings.insert(name.clone(), InitializerBinding::SrcField(field.clone()));
                }
            }
        }
        Ok(bindings)
    }

    fn datum_expression_bindings(
        &self,
        type_path: &TypePath,
    ) -> Result<BTreeMap<String, InitializerBinding>, RuntimeImageError> {
        let mut bindings = self
            .binding_index
            .globals
            .iter()
            .map(|(name, field)| (name.clone(), InitializerBinding::Global(field.clone())))
            .collect::<BTreeMap<_, _>>();
        for builtin in ["type", "parent_type"] {
            let field = FieldName::parse(builtin).expect("built-in datum field is valid");
            bindings.insert(builtin.to_owned(), InitializerBinding::SrcField(field));
        }
        let mut owners = Vec::new();
        let mut current = Some(type_path.clone());
        let mut visited = BTreeSet::new();
        while let Some(path) = current.take() {
            if !visited.insert(path.clone()) {
                return Err(RuntimeImageError::InheritanceCycle(path));
            }
            let runtime_type = self
                .types
                .get(&path)
                .ok_or_else(|| RuntimeImageError::UnknownType(path.clone()))?;
            owners.push(path);
            current.clone_from(&runtime_type.parent);
        }
        owners.reverse();
        for owner in owners {
            if let Some(fields) = self.binding_index.instance_fields.get(owner.as_str()) {
                for (name, field) in fields {
                    bindings.insert(name.clone(), InitializerBinding::SrcField(field.clone()));
                }
            }
        }
        Ok(bindings)
    }

    fn default_layers(
        &self,
        type_path: &TypePath,
    ) -> Result<Vec<DatumDefaults>, DynamicInitializerFailure> {
        let mut layers = Vec::new();
        let mut current = Some(type_path.clone());
        let mut visited = BTreeSet::new();
        while let Some(path) = current.take() {
            if !visited.insert(path.clone()) {
                return Err(DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Execution,
                    message: format!("runtime inheritance cycle at {path}"),
                    expanded_span: SourceSpan::new(0, 0),
                });
            }
            let runtime_type = self
                .types
                .get(&path)
                .ok_or_else(|| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Execution,
                    message: format!("runtime type {path} is absent"),
                    expanded_span: SourceSpan::new(0, 0),
                })?;
            layers.push(runtime_type.defaults.clone());
            current.clone_from(&runtime_type.parent);
        }
        layers.reverse();
        Ok(layers)
    }

    fn retain_dynamic_failure(
        &mut self,
        compilation: &Compilation,
        entry: &VariableEntry,
        step: &InitializationStep,
        unsupported: &dm_globals::UnsupportedConstant,
        failure: DynamicInitializerFailure,
    ) -> Result<(), RuntimeImageError> {
        let initializer = entry
            .initializer
            .as_ref()
            .ok_or_else(|| RuntimeImageError::MissingInitializer(step.path.clone()))?;
        let file = compilation
            .project()
            .file(entry.file_id)
            .ok_or(RuntimeImageError::MissingSourceFile(entry.file_id))?;
        let blocker_span = compilation
            .original_span(entry.file_id, failure.expanded_span)
            .ok_or(RuntimeImageError::MissingSourceFile(entry.file_id))?;
        self.diagnostics.push(RuntimeInitializerDiagnostic {
            variable_path: step.path.clone(),
            storage: step.storage,
            ordinal: step.ordinal,
            file_id: entry.file_id,
            source_path: file.relative_path.to_string_lossy().into_owned(),
            initializer_span: initializer.original_span,
            blocker_span,
            category: unsupported.category,
            phase: failure.phase,
            message: failure.message,
        });
        Ok(())
    }
}

fn runtime_types(
    compilation: &Compilation,
) -> Result<BTreeMap<TypePath, RuntimeType>, RuntimeImageError> {
    let mut types = BTreeMap::new();
    for node in compilation.code_tree().nodes() {
        if node.kind != NodeKind::Type {
            continue;
        }
        let path = parse_type_path(&node.path.to_string())?;
        let parent = node
            .parent_type
            .and_then(|parent| compilation.code_tree().node(parent))
            .map(|parent| parse_type_path(&parent.path.to_string()))
            .transpose()?;
        types.insert(
            path.clone(),
            RuntimeType {
                defaults: DatumDefaults::new(path.clone()),
                path,
                parent,
            },
        );
    }
    Ok(types)
}

fn parse_type_path(path: &str) -> Result<TypePath, RuntimeImageError> {
    TypePath::parse(path).map_err(RuntimeImageError::Value)
}

fn variable_field(path: &str) -> Result<FieldName, RuntimeImageError> {
    let name = path
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| RuntimeImageError::InvalidVariablePath(path.to_owned()))?;
    FieldName::parse(name).map_err(RuntimeImageError::Value)
}

/// Failure while converting a valid frontend snapshot into runtime storage.
#[derive(Debug)]
pub enum RuntimeImageError {
    /// Runtime canonical-value validation failed.
    Value(ValueError),
    /// A planned variable path had no final field segment.
    InvalidVariablePath(String),
    /// A type referenced by a plan or allocation was absent.
    UnknownType(TypePath),
    /// Retained type metadata contained an inheritance cycle.
    InheritanceCycle(TypePath),
    /// An initialization plan referred to a missing initializer.
    MissingInitializer(String),
    /// An instance-default plan referred to a variable without an owner.
    MissingOwner(String),
    /// A retained dynamic instance initializer failed for one allocated datum.
    InstanceInitializer {
        /// Canonical variable path being initialized.
        path: String,
        /// Lowering or execution detail.
        message: String,
    },
    /// An initialization entry referred to an absent project file.
    MissingSourceFile(FileId),
    /// A raw live-datum expression could not be lexed or lowered.
    ExpressionLowering {
        /// Recoverable compiler detail.
        message: String,
        /// Expression-relative source span associated with the failure.
        source_span: SourceSpan,
    },
    /// A raw live-datum expression failed in the VM.
    ExpressionExecution(RuntimeError),
}

impl fmt::Display for RuntimeImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(error) => write!(formatter, "invalid runtime value: {error}"),
            Self::InvalidVariablePath(path) => write!(formatter, "invalid variable path {path:?}"),
            Self::UnknownType(path) => write!(formatter, "runtime type {path} is absent"),
            Self::InheritanceCycle(path) => {
                write!(formatter, "runtime inheritance cycle at {path}")
            }
            Self::MissingInitializer(path) => {
                write!(
                    formatter,
                    "initialization step for {path} has no initializer"
                )
            }
            Self::MissingOwner(path) => {
                write!(formatter, "instance variable {path} has no owning type")
            }
            Self::InstanceInitializer { path, message } => {
                write!(
                    formatter,
                    "instance initializer for {path} failed: {message}"
                )
            }
            Self::MissingSourceFile(file) => {
                write!(
                    formatter,
                    "initializer source file {} is absent",
                    file.index()
                )
            }
            Self::ExpressionLowering {
                message,
                source_span,
            } => write!(
                formatter,
                "expression lowering failed at source {}..{}: {message}",
                source_span.start, source_span.end
            ),
            Self::ExpressionExecution(error) => {
                write!(formatter, "expression execution failed: {error}")
            }
        }
    }
}

impl std::error::Error for RuntimeImageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Value(error) => Some(error),
            Self::InvalidVariablePath(_)
            | Self::UnknownType(_)
            | Self::InheritanceCycle(_)
            | Self::MissingInitializer(_)
            | Self::MissingOwner(_)
            | Self::InstanceInitializer { .. }
            | Self::MissingSourceFile(_)
            | Self::ExpressionLowering { .. } => None,
            Self::ExpressionExecution(error) => Some(error),
        }
    }
}

impl From<ValueError> for RuntimeImageError {
    fn from(error: ValueError) -> Self {
        Self::Value(error)
    }
}

/// Failure while loading or materializing a project.
#[derive(Debug)]
pub enum RuntimeImageLoadError {
    /// Frontend project loading failed.
    Compiler(CompilerError),
    /// Runtime materialization failed.
    Image(RuntimeImageError),
}

impl fmt::Display for RuntimeImageLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiler(error) => write!(formatter, "frontend compilation failed: {error}"),
            Self::Image(error) => write!(formatter, "runtime image failed: {error}"),
        }
    }
}

impl std::error::Error for RuntimeImageLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compiler(error) => Some(error),
            Self::Image(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::CompilerDatabase;
    use dm_globals::{StorageClass, UnsupportedCategory};
    use dm_value::{FieldName, TypePath, Value};

    use super::{
        ConstantFieldApplication, InitializerFailurePhase, RuntimeImage, RuntimeImageError,
    };

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "dream64-dm-runtime-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("fixture directory should be created");
            Self(path)
        }

        fn write(&self, name: &str, source: &str) {
            fs::write(self.0.join(name), source).expect("fixture source should be written");
        }

        fn image(&self) -> RuntimeImage {
            let compilation = CompilerDatabase::new()
                .compile(self.0.join("world.dme"))
                .expect("fixture should compile");
            RuntimeImage::from_compilation(&compilation).expect("fixture should materialize")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("fixture directory should be removed");
        }
    }

    fn field(name: &str) -> FieldName {
        FieldName::parse(name).expect("test field should be valid")
    }

    fn type_path(path: &str) -> TypePath {
        TypePath::parse(path).expect("test type path should be valid")
    }

    #[test]
    fn applies_global_and_static_overrides_in_project_order() {
        let fixture = Fixture::new();
        fixture.write(
            "world.dme",
            "#include \"first.dm\"\n#include \"second.dm\"\n",
        );
        fixture.write(
            "first.dm",
            "var/global/root = 1\n/datum/example\n\tvar/static/shared = 2\n",
        );
        fixture.write("second.dm", "root = 3\n/datum/example\n\tshared = 4\n");

        let image = fixture.image();
        let root = image
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with("/root"))
            .expect("root should be materialized");
        let shared = image
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with("/shared"))
            .expect("static should be materialized");

        assert_eq!(root.storage, StorageClass::Global);
        assert_eq!(root.value.as_number(), Some(3.0));
        assert_eq!(shared.storage, StorageClass::Static);
        assert_eq!(shared.value.as_number(), Some(4.0));
        assert!(root.ordinal < shared.ordinal);
        assert_eq!(image.stats().constants_materialized, 4);
        assert_eq!(image.stats().runtime_variables, 2);
    }

    #[test]
    fn type_statics_use_distinct_qualified_persistent_vm_slots() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/a\n\tvar/static/shared = 1\n/datum/b\n\tvar/static/shared = 2\n",
        );
        let mut image = fixture.image();
        let a_path = image
            .variables()
            .iter()
            .find(|variable| {
                variable.path.contains("/datum/a/") && variable.path.ends_with("/shared")
            })
            .expect("a static")
            .path
            .clone();
        let b_path = image
            .variables()
            .iter()
            .find(|variable| {
                variable.path.contains("/datum/b/") && variable.path.ends_with("/shared")
            })
            .expect("b static")
            .path
            .clone();
        let a_slot = FieldName::static_storage(&a_path);
        let b_slot = FieldName::static_storage(&b_path);
        assert_ne!(a_slot, b_slot);
        let mut state = image.take_execution_state();
        assert_eq!(state.global(&a_slot).and_then(Value::as_number), Some(1.0));
        assert_eq!(state.global(&b_slot).and_then(Value::as_number), Some(2.0));
        state.set_global(a_slot, Value::number(7.0));
        image.restore_execution_state(state);
        assert_eq!(
            image
                .variable(&a_path)
                .and_then(|value| value.value.as_number()),
            Some(7.0)
        );
        assert_eq!(
            image
                .variable(&b_path)
                .and_then(|value| value.value.as_number()),
            Some(2.0)
        );
    }

    #[test]
    fn dynamic_global_datum_initializer_is_materialized_and_visible_to_later_steps() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/item\n\tvar/value = 8\n/var/global/datum/item/first = new /datum/item\n/var/global/second = first\n",
        );
        let image = fixture.image();
        let first = image.variable("/var/first").expect("first global");
        let second = image.variable("/var/second").expect("second global");
        assert!(matches!(first.value, Value::Datum(_)));
        assert_eq!(second.value, first.value);
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
    }

    #[test]
    fn dynamic_global_datum_has_plain_declared_null_fields() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/var/global/datum/log_holder/logger = new /datum/log_holder\n/datum/log_holder\n\tvar/list/waiting_log_calls\n\tvar/initialized = FALSE\n",
        );
        let image = fixture.image();
        let Value::Datum(logger) = &image.variable("/var/logger").expect("logger global").value
        else {
            panic!("logger should be a datum");
        };
        let logger = image.heap().datum(*logger).expect("logger should be live");
        assert_eq!(
            logger.field(&field("waiting_log_calls")),
            Ok(&Value::Null),
            "an uninitialized declaration is still a real null-valued field"
        );
        assert_eq!(logger.field(&field("initialized")), Ok(&Value::number(0.0)));
    }

    #[test]
    fn dynamic_type_static_datum_initializer_uses_qualified_storage() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/item\n/datum/holder\n\tvar/static/datum/item/shared = new /datum/item\n",
        );
        let mut image = fixture.image();
        let variable = image
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with("/shared"))
            .expect("static")
            .clone();
        assert!(matches!(variable.value, Value::Datum(_)));
        let state = image.take_execution_state();
        assert_eq!(
            state.global(&FieldName::static_storage(&variable.path)),
            Some(&variable.value)
        );
        assert!(image.diagnostics().is_empty());
    }

    #[test]
    fn dynamic_initializer_calls_project_procedure_through_one_linked_module() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/proc/build_value(var/base)\n\treturn base + 5\n/var/global/seed = 7\n/var/global/result = build_value(seed)\n",
        );
        let image = fixture.image();
        assert_eq!(
            image
                .variable("/var/result")
                .and_then(|v| v.value.as_number()),
            Some(12.0),
            "vars={:?} diagnostics={:?}",
            image.variables(),
            image.diagnostics()
        );
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
    }

    #[test]
    fn world_singleton_exists_during_global_initialization() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "/var/global/start_tick_lag = world.tick_lag\n");
        let mut image = fixture.image();
        let world = image.canonical_world().expect("world is preallocated");
        assert_eq!(
            image
                .variable("/var/start_tick_lag")
                .and_then(|v| v.value.as_number()),
            Some(1.0)
        );
        let state = image.take_execution_state();
        assert_eq!(state.global(&field("world")), Some(&Value::Datum(world)));
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
    }

    #[test]
    fn layers_ancestor_defaults_and_reopen_overrides_deterministically() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/name = \"base\"\n\tvar/health = 10\n/datum/base/child\n\tname = \"child\"\n/datum/base/child\n\tname = \"reopened\"\n\tvar/speed = 2\n",
        );

        let mut image = fixture.image();
        let datum_id = image
            .allocate_datum(&type_path("/datum/base/child"))
            .expect("child datum should allocate");
        let datum = image.heap().datum(datum_id).expect("datum should be live");
        let names = datum
            .fields()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, ["name", "health", "speed", "tag"]);
        assert_eq!(datum.field(&field("name")), Ok(&Value::text("reopened")));
        assert_eq!(
            datum.field(&field("health")).unwrap().as_number(),
            Some(10.0)
        );
        assert_eq!(datum.field(&field("speed")).unwrap().as_number(), Some(2.0));
        assert_eq!(image.stats().default_layers, 2);
    }

    #[test]
    fn implicit_new_override_uses_the_inherited_declared_field_type() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/test/thing\n\tvar/list/foo = list()\n/datum/test/thing/stuff\n\tfoo = new()\n",
        );

        let mut image = fixture.image();
        let datum_id = image
            .allocate_datum(&type_path("/datum/test/thing/stuff"))
            .expect("subtype datum should allocate");
        let datum = image.heap().datum(datum_id).expect("datum should be live");
        assert!(matches!(
            datum
                .fields()
                .find(|(name, _)| *name == &field("foo"))
                .map(|(_, value)| value),
            Some(Value::List(_))
        ));
    }

    #[test]
    fn world_params_is_an_empty_indexable_list_without_host_parameters() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "/world\n");
        let mut image = fixture.image();
        let world = image
            .allocate_datum(&type_path("/world"))
            .expect("world should allocate");

        let params = image
            .heap()
            .datum_field(world, &field("params"))
            .expect("world.params exists");
        let Value::List(params) = params else {
            panic!("world.params should be an indexable empty list");
        };
        assert_eq!(image.heap().list(*params).expect("params list").len(), 0);
        assert_eq!(
            image.heap().datum_field(world, &field("log")),
            Ok(&Value::Null),
            "world.log exists before the project selects an output sink"
        );
        assert_eq!(
            image.heap().datum_field(world, &field("internet_address")),
            Ok(&Value::Null)
        );
        assert_eq!(
            image.heap().datum_field(world, &field("area")),
            Ok(&Value::TypePath(type_path("/area")))
        );
        assert_eq!(
            image.heap().datum_field(world, &field("byond_version")),
            Ok(&Value::number(516.0))
        );
        assert_eq!(
            image.heap().datum_field(world, &field("tick_usage")),
            Ok(&Value::number(0.0))
        );
        let state = image.take_execution_state();
        assert_eq!(
            state.initial_value(&type_path("/world"), &field("params")),
            Some(&Value::Null),
            "initial() and dynamic world construction must observe the same built-in default"
        );
        for nullable in [
            "hub",
            "hub_password",
            "internet_address",
            "address",
            "status",
        ] {
            assert_eq!(
                state.initial_value(&type_path("/world"), &field(nullable)),
                Some(&Value::Null),
                "documented host value {nullable} defaults to null"
            );
        }
        assert_eq!(
            state.initial_value(&type_path("/world"), &field("port")),
            Some(&Value::number(0.0))
        );
        assert_eq!(
            state.initial_value(&type_path("/world"), &field("name")),
            Some(&Value::text("world")),
            "the default world name is the environment file stem"
        );
        for (name, expected) in [
            ("area", Value::TypePath(type_path("/area"))),
            ("mob", Value::TypePath(type_path("/mob"))),
            ("turf", Value::TypePath(type_path("/turf"))),
            ("byond_version", Value::number(516.0)),
            ("byond_build", Value::number(1663.0)),
            ("cache_lifespan", Value::number(30.0)),
            ("game_state", Value::number(0.0)),
            ("loop_checks", Value::number(1.0)),
            ("map_format", Value::number(0.0)),
            ("map_cpu", Value::number(0.0)),
            ("movement_mode", Value::number(0.0)),
            ("reachable", Value::number(0.0)),
            ("sleep_offline", Value::number(0.0)),
            ("tick_usage", Value::number(0.0)),
            ("version", Value::number(0.0)),
            ("view", Value::number(5.0)),
            ("visibility", Value::number(1.0)),
        ] {
            assert_eq!(
                state.initial_value(&type_path("/world"), &field(name)),
                Some(&expected),
                "live and initial metadata disagree for {name}"
            );
        }
    }

    #[test]
    fn instance_initializer_can_read_an_implicit_parent_type_field() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/flags = 4\n/datum/base/child\n\tflags = parent_type::flags | 2\n",
        );

        let mut image = fixture.image();
        let datum_id = image
            .allocate_datum(&type_path("/datum/base/child"))
            .expect("parent_type initializer should allocate");
        let datum = image.heap().datum(datum_id).expect("datum should be live");
        assert_eq!(datum.field(&field("flags")).unwrap().as_number(), Some(6.0));
    }

    #[test]
    fn instance_initializer_preserves_bare_associative_list_keys_as_text() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/fax\n\tvar/fax_name = \"field collision\"\n\tvar/list/networks = list(nanotrasen = list(fax_name = \"NT HR\"), syndicate = list(fax_name = \"Sabotage\"))\n",
        );

        let mut image = fixture.image();
        let datum_id = image
            .allocate_datum(&type_path("/datum/fax"))
            .expect("associative initializer should allocate");
        let Value::List(networks) = image
            .heap()
            .datum(datum_id)
            .unwrap()
            .field(&field("networks"))
            .unwrap()
            .clone()
        else {
            panic!("networks should be a list");
        };
        assert!(
            image
                .heap()
                .list(networks)
                .unwrap()
                .associations()
                .any(|(key, _)| key == &Value::text("nanotrasen"))
        );
    }

    #[test]
    fn macro_generated_semicolon_assignments_have_distinct_initializers() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "#define SET_PIXELS(x, y) pixel_x = x; base_pixel_x = x; pixel_y = y; base_pixel_y = y;\n/obj/canvas\n\tSET_PIXELS(11, 10)\n",
        );

        let mut image = fixture.image();
        let datum_id = image
            .allocate_datum(&type_path("/obj/canvas"))
            .expect("semicolon-separated macro fields should allocate");
        let datum = image.heap().datum(datum_id).unwrap();
        for (name, expected) in [
            ("pixel_x", 11.0),
            ("base_pixel_x", 11.0),
            ("pixel_y", 10.0),
            ("base_pixel_y", 10.0),
        ] {
            assert_eq!(
                datum.field(&field(name)).unwrap().as_number(),
                Some(expected),
                "{name} should retain its own initializer"
            );
        }
    }

    #[test]
    fn evaluates_dynamic_instance_defaults_per_datum_in_inherited_source_order() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/list/items = list(1)\n\tvar/datum/base/child = new /datum/base\n/datum/base/sub\n\titems = list(2)\n\tchild = new /datum/base/sub\n",
        );

        let mut image = fixture.image();
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
        let first = image
            .allocate_datum(&type_path("/datum/base/sub"))
            .expect("first subtype should allocate");
        let second = image
            .allocate_datum(&type_path("/datum/base/sub"))
            .expect("second subtype should allocate");
        assert_eq!(
            image.stats().instance_initializer_plans_compiled,
            1,
            "repeated allocations of one type must reuse its compiled initializer plan"
        );
        assert_eq!(
            image.stats().datum_allocation_plans_built,
            1,
            "repeated allocations must reuse inherited defaults and ancestry metadata"
        );

        let read = |image: &RuntimeImage, datum, name: &str| {
            image
                .heap()
                .datum(datum)
                .expect("datum should be live")
                .fields()
                .find(|(field_name, _)| *field_name == &field(name))
                .map(|(_, value)| value.clone())
                .expect("dynamic field should be initialized")
        };
        let (Value::List(first_items), Value::List(second_items)) =
            (read(&image, first, "items"), read(&image, second, "items"))
        else {
            panic!("items defaults should be lists");
        };
        assert_ne!(first_items, second_items, "instance lists must not alias");
        assert_eq!(
            image
                .heap()
                .list(first_items)
                .unwrap()
                .positions()
                .next()
                .map(|(_, value)| value),
            Some(&Value::number(2.0)),
            "subtype override must run after the inherited initializer"
        );

        let (Value::Datum(first_child), Value::Datum(second_child)) =
            (read(&image, first, "child"), read(&image, second, "child"))
        else {
            panic!("child defaults should be datums");
        };
        assert_ne!(first_child, second_child, "instance datums must not alias");
        assert_eq!(
            image
                .heap()
                .datum(first_child)
                .unwrap()
                .type_path()
                .as_str(),
            "/datum/base/sub"
        );
    }

    #[test]
    fn preflights_unique_initializer_plans_without_allocating_datums() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/list/items = list(1)\n/datum/base/sub\n\titems = list(2)\n",
        );
        let mut image = fixture.image();
        let subtype = type_path("/datum/base/sub");
        let stats = image
            .preflight_instance_initializers([subtype.clone(), subtype.clone()])
            .expect("valid plans should preflight");
        assert_eq!(stats.types, 1);
        assert_eq!(stats.plans_compiled, 1);
        assert_eq!(stats.plans_reused, 0);
        assert_eq!(image.stats().datums_allocated, 0);

        let reused = image
            .preflight_instance_initializers([subtype])
            .expect("cached plan should preflight");
        assert_eq!(reused.plans_compiled, 0);
        assert_eq!(reused.plans_reused, 1);
        assert_eq!(image.stats().datums_allocated, 0);
    }

    #[test]
    fn preflight_aggregates_failures_in_canonical_type_order() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/a\n\tvar/value = missing_a()\n/datum/b\n\tvar/value = missing_b()\n",
        );
        let mut image = fixture.image();
        let errors = image
            .preflight_instance_initializers([type_path("/datum/b"), type_path("/datum/a")])
            .expect_err("invalid plans should fail preflight");
        assert_eq!(errors.len(), 2);
        assert!(errors[0].to_string().contains("/datum/a"));
        assert!(errors[1].to_string().contains("/datum/b"));
        assert_eq!(image.stats().datums_allocated, 0);
    }

    #[test]
    fn materializes_contextual_upward_path_type_defaults() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write("types.dm", "/datum/foo\n/datum/bar\n\tvar/meep = .foo\n");
        let mut image = fixture.image();
        let datum = image
            .allocate_datum(&type_path("/datum/bar"))
            .expect("bar should allocate");

        assert_eq!(
            image.heap().datum_field(datum, &field("meep")),
            Ok(&Value::TypePath(type_path("/datum/foo")))
        );
    }

    #[test]
    fn materializes_builtin_atom_appearance_defaults_without_overriding_source() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write("types.dm", "/obj/example\n\talpha = 127\n");

        let mut image = fixture.image();
        let datum_id = image
            .allocate_datum(&type_path("/obj/example"))
            .expect("atom subtype should allocate");
        let datum = image.heap().datum(datum_id).expect("datum should be live");
        assert_eq!(
            datum.field(&field("alpha")).unwrap().as_number(),
            Some(127.0)
        );
        assert_eq!(
            datum.field(&field("appearance_flags")).unwrap().as_number(),
            Some(0.0)
        );
        assert_eq!(datum.field(&field("layer")).unwrap().as_number(), Some(1.0));
        assert_eq!(datum.field(&field("plane")).unwrap().as_number(), Some(0.0));
        assert_eq!(datum.field(&field("transform")), Ok(&Value::Null));
    }

    #[test]
    fn converts_nested_associative_lists_per_instance() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"lists.dm\"\n");
        fixture.write(
            "lists.dm",
            "/datum/holder\n\tvar/items = list(1, \"nested\" = list(2, \"answer\" = 3))\n",
        );

        let mut image = fixture.image();
        let repeated = fixture.image();
        assert_eq!(image.variables, repeated.variables);
        assert_eq!(image.types, repeated.types);
        assert_eq!(image.diagnostics, repeated.diagnostics);
        assert_eq!(image.stats, repeated.stats);

        let holder = type_path("/datum/holder");
        let first = image
            .allocate_datum(&holder)
            .expect("first holder should allocate");
        let second = image
            .allocate_datum(&holder)
            .expect("second holder should allocate");
        let first_items = image
            .heap()
            .datum(first)
            .unwrap()
            .field(&field("items"))
            .unwrap()
            .clone();
        let second_items = image
            .heap()
            .datum(second)
            .unwrap()
            .field(&field("items"))
            .unwrap()
            .clone();
        let (Value::List(first_list), Value::List(second_list)) = (first_items, second_items)
        else {
            panic!("items should be list handles");
        };

        assert_ne!(
            first_list, second_list,
            "mutable instance list defaults must not alias"
        );
        let outer = image.heap().list(first_list).unwrap();
        assert_eq!(outer.get(1).unwrap().as_number(), Some(1.0));
        let Value::List(nested_id) = outer.get_key(&Value::text("nested")).unwrap() else {
            panic!("nested key should contain a list handle");
        };
        let nested = image.heap().list(*nested_id).unwrap();
        assert_eq!(nested.get(1).unwrap().as_number(), Some(2.0));
        assert_eq!(
            nested.get_key(&Value::text("answer")).unwrap().as_number(),
            Some(3.0)
        );
        assert_eq!(image.stats().constant_lists, 0);
    }

    #[test]
    fn retains_source_mapped_unsupported_initializers_without_guessing() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/dynamic = build_value()\n/datum/example\n\tvar/runtime = new /datum\n",
        );

        let mut image = fixture.image();
        let diagnostics = image.diagnostics();

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].category, UnsupportedCategory::Call);
        assert_eq!(diagnostics[0].phase, InitializerFailurePhase::Lowering);
        assert!(diagnostics[0].message.contains("unknown procedure"));
        assert!(
            diagnostics
                .iter()
                .all(|diagnostic| diagnostic.source_path == "vars.dm")
        );
        assert!(diagnostics.iter().all(|diagnostic| {
            !diagnostic.initializer_span.is_empty() && !diagnostic.blocker_span.is_empty()
        }));
        let datum = image
            .allocate_datum(&type_path("/datum/example"))
            .expect("deferred instance initializer should execute on allocation");
        assert!(matches!(
            image.heap().datum_field(datum, &field("runtime")),
            Ok(Value::Datum(_))
        ));
        assert!(
            image
                .variables()
                .iter()
                .all(|variable| !variable.path.ends_with("/dynamic"))
        );

        let datum_id = image
            .allocate_datum(&type_path("/datum/example"))
            .expect("datum should still allocate");
        let first_runtime = image
            .heap()
            .datum_field(datum, &field("runtime"))
            .unwrap()
            .clone();
        let second_runtime = image
            .heap()
            .datum_field(datum_id, &field("runtime"))
            .unwrap()
            .clone();
        assert!(matches!(second_runtime, Value::Datum(_)));
        assert_ne!(
            first_runtime, second_runtime,
            "instance datums must not alias"
        );
    }

    #[test]
    fn uninitialized_instance_fields_materialize_as_null_on_new_datums() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/inherited_missing\n\tvar/inherited_value = 4\n/datum/base/log_holder\n\tvar/list/waiting_log_calls\n\tvar/list/data_cache = list()\n",
        );
        let mut image = fixture.image();
        let datum = image
            .allocate_datum(&type_path("/datum/base/log_holder"))
            .expect("log holder should allocate with every declared field");

        assert_eq!(
            image.heap().datum_field(datum, &field("waiting_log_calls")),
            Ok(&Value::Null)
        );
        assert_eq!(
            image.heap().datum_field(datum, &field("inherited_missing")),
            Ok(&Value::Null)
        );
        assert_eq!(
            image.heap().datum_field(datum, &field("inherited_value")),
            Ok(&Value::number(4.0))
        );
        assert!(matches!(
            image.heap().datum_field(datum, &field("data_cache")),
            Ok(Value::List(_))
        ));
    }

    #[test]
    fn executes_identifier_dependencies_and_overrides_in_source_order() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/base = 2\nvar/global/derived = base + 3\nbase = 10\nvar/global/final_value = base + derived\n",
        );

        let image = fixture.image();
        let number = |suffix: &str| {
            image
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with(suffix))
                .and_then(|variable| variable.value.as_number())
        };

        assert_eq!(number("/base"), Some(10.0));
        assert_eq!(number("/derived"), Some(5.0));
        assert_eq!(number("/final_value"), Some(15.0));
        assert_eq!(image.stats().constants_materialized, 2);
        assert_eq!(image.stats().dynamic_initializers_materialized, 2);
        assert!(image.diagnostics().is_empty());
    }

    #[test]
    fn materializes_min_max_global_initializers() {
        let fixture = Fixture::new();
        fixture.write(
            "world.dme",
            "var/global/tick_limit = max(100, 80 + 5)\nvar/global/lower = min(list(7, 3, 9))\n",
        );
        let image = fixture.image();
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
        assert_eq!(
            image
                .variables()
                .iter()
                .find(|value| value.path.ends_with("/tick_limit"))
                .map(|value| &value.value),
            Some(&Value::number(100.0))
        );
        assert_eq!(
            image
                .variables()
                .iter()
                .find(|value| value.path.ends_with("/lower"))
                .map(|value| &value.value),
            Some(&Value::number(3.0))
        );
    }

    #[test]
    fn dynamic_initializer_global_writes_persist_into_later_steps() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/base = 1\nvar/global/assigned = (base = base + 2)\nvar/global/observed = base\n",
        );

        let image = fixture.image();
        let number = |suffix: &str| {
            image
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with(suffix))
                .and_then(|variable| variable.value.as_number())
        };
        assert_eq!(number("/base"), Some(3.0));
        assert_eq!(number("/assigned"), Some(3.0));
        assert_eq!(number("/observed"), Some(3.0));
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
    }

    #[test]
    fn executes_src_field_and_explicit_global_references() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/offset = 3\n/datum/example\n\tvar/base = 4\n\tvar/combined = base + global.offset\n",
        );

        let mut image = fixture.image();
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
        let datum = image
            .allocate_datum(&type_path("/datum/example"))
            .expect("datum should allocate");

        assert_eq!(
            image
                .heap()
                .datum_field(datum, &field("combined"))
                .unwrap()
                .as_number(),
            Some(7.0)
        );
        assert_eq!(image.stats().dynamic_initializers_materialized, 1);
        assert!(image.diagnostics().is_empty());
    }

    #[test]
    fn executes_list_expressions_with_runtime_values() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/seed = 2\nvar/global/items = list(seed, \"answer\" = seed + 1)\n",
        );

        let image = fixture.image();
        let items = image
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with("/items"))
            .expect("items should materialize");
        let Value::List(items) = items.value else {
            panic!("items should be a runtime list");
        };
        let list = image.heap().list(items).expect("list should remain live");

        assert_eq!(list.get(1).unwrap().as_number(), Some(2.0));
        assert_eq!(
            list.get_key(&Value::text("answer")).unwrap().as_number(),
            Some(3.0)
        );
        assert_eq!(image.stats().dynamic_initializers_materialized, 1);
        assert!(image.diagnostics().is_empty());
    }

    #[test]
    fn applies_only_proven_constant_field_expressions() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write("types.dm", "/datum/example\n\tvar/value = 2\n");
        let mut image = fixture.image();
        let datum = image
            .allocate_datum(&type_path("/datum/example"))
            .expect("datum should allocate");

        assert_eq!(
            image
                .apply_constant_field_expression(datum, field("value"), "3 + 4")
                .expect("constant should apply"),
            ConstantFieldApplication::Applied
        );
        let unsupported = image
            .apply_constant_field_expression(datum, field("value"), "build_value()")
            .expect("unsupported expression should remain recoverable");
        assert!(matches!(
            unsupported,
            ConstantFieldApplication::Unsupported(ref blocker)
                if blocker.category == UnsupportedCategory::Call
        ));
        assert_eq!(
            image
                .heap()
                .datum_field(datum, &field("value"))
                .unwrap()
                .as_number(),
            Some(7.0)
        );
        assert_eq!(image.stats().datums_allocated, 1);
    }

    #[test]
    fn evaluates_live_datum_expressions_with_materialized_globals() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "var/global/offset = 3\n/datum/base\n\tvar/base = 4\n/datum/base/child\n\tvar/value = 0\n",
        );
        let mut image = fixture.image();
        let datum = image
            .allocate_datum(&type_path("/datum/base/child"))
            .expect("child datum should allocate");
        let mut state = image.take_execution_state();

        assert_eq!(
            image
                .evaluate_datum_expression(datum, "base + global.offset", &mut state)
                .expect("expression should execute")
                .as_number(),
            Some(7.0)
        );
        state
            .heap_mut()
            .set_datum_field(datum, field("base"), Value::number(9.0))
            .expect("datum should remain live in execution state");
        assert_eq!(
            image
                .evaluate_datum_expression(datum, "base + offset", &mut state)
                .expect("bare global should execute")
                .as_number(),
            Some(12.0)
        );
        state
            .heap_mut()
            .delete_datum_field(datum, &field("base"))
            .expect("datum should remain live in execution state");
        let error = image
            .evaluate_datum_expression(datum, "base", &mut state)
            .expect_err("missing field should retain VM failure details");
        assert!(matches!(
            error,
            RuntimeImageError::ExpressionExecution(ref error)
                if error.source_span.is_some() && !error.call_stack.is_empty()
        ));
    }

    #[test]
    fn execution_state_carries_initial_parent_and_project_metadata() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/value = 7\n/datum/base/child\n",
        );
        let mut image = fixture.image();
        let state = image.take_execution_state();
        let child = type_path("/datum/base/child");
        assert_eq!(state.type_parent(&child), Some(&type_path("/datum/base")));
        assert_eq!(
            state.initial_value(&child, &field("value")),
            Some(&Value::number(7.0))
        );
        let project_root = state
            .project_root()
            .expect("image should retain its project root");
        assert_eq!(
            std::fs::canonicalize(project_root).expect("project root should exist"),
            std::fs::canonicalize(&fixture.0).expect("fixture root should exist")
        );
    }

    #[test]
    fn execution_states_share_the_image_type_catalog() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write("types.dm", "/datum/child\n/obj/item\n");
        let mut image = fixture.image();

        assert_eq!(Arc::strong_count(&image.type_paths), 1);
        let state = image.take_execution_state();
        assert_eq!(Arc::strong_count(&image.type_paths), 2);
        drop(state);
        assert_eq!(Arc::strong_count(&image.type_paths), 1);
    }

    #[test]
    fn repeated_execution_state_transfers_reuse_cached_type_metadata() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "var/global/counter = 1\n/datum/base\n\tvar/value = 7\n/datum/base/child\n\tvar/list/items = list(value)\n",
        );
        let mut image = fixture.image();
        let builds = image.stats().execution_metadata_builds;
        assert_eq!(Arc::strong_count(&image.type_parents), 1);
        assert_eq!(Arc::strong_count(&image.initial_values), 1);

        for _ in 0..3 {
            let state = image.take_execution_state();
            assert_eq!(Arc::strong_count(&image.type_parents), 2);
            assert_eq!(Arc::strong_count(&image.initial_values), 2);
            image.restore_execution_state(state);
        }
        assert_eq!(image.stats().execution_metadata_builds, builds);

        image
            .allocate_datum(&type_path("/datum/base/child"))
            .expect("dynamic defaults should allocate");
        image
            .allocate_datum(&type_path("/datum/base/child"))
            .expect("repeated dynamic defaults should allocate");
        assert_eq!(image.stats().execution_metadata_builds, builds);
    }
}
