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
use std::time::{Duration, Instant};

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
    ExecutionContext, ExecutionState, InitializerBinding, InitializerProgram, InstanceInitializer,
    Module, RuntimeError, compile_initializer, compile_initializer_into_module,
    execute_module_in_context,
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

fn builtin_mob_defaults() -> [(&'static str, Value); 9] {
    [
        ("client", Value::Null),
        ("eye", Value::Null),
        ("key", Value::Null),
        ("ckey", Value::Null),
        ("perspective", Value::number(0.0)),
        ("see_in_dark", Value::number(2.0)),
        ("see_infrared", Value::number(0.0)),
        ("see_invisible", Value::number(0.0)),
        ("sight", Value::number(0.0)),
    ]
}

fn builtin_client_defaults() -> [(&'static str, Value); 12] {
    [
        ("control_freak", Value::number(0.0)),
        ("dir", Value::number(2.0)),
        ("eye", Value::Null),
        ("gender", Value::text("neuter")),
        ("inactivity", Value::number(0.0)),
        ("mob", Value::Null),
        ("perspective", Value::number(0.0)),
        ("pixel_x", Value::number(0.0)),
        ("pixel_y", Value::number(0.0)),
        ("pixel_z", Value::number(0.0)),
        ("pixel_w", Value::number(0.0)),
        ("statobj", Value::Null),
    ]
}

fn materialize_builtin_atom_defaults(
    heap: &mut ValueHeap,
    datum: DatumId,
    is_atom: bool,
    is_movable: bool,
    is_mob: bool,
    is_client: bool,
    is_particles: bool,
    is_image: bool,
) -> Result<(), ValueError> {
    // Every /datum has BYOND's built-in tag field, even though it has no
    // source declaration in user projects.
    let tag = FieldName::parse("tag").expect("built-in datum field name is valid");
    if heap.datum_field(datum, &tag).is_err() {
        heap.set_datum_field(datum, tag, Value::Null)?;
    }
    if is_particles {
        for (name, value) in particle_defaults() {
            let field = FieldName::parse(name).expect("built-in particle field");
            if heap.datum_field(datum, &field).is_err() {
                heap.set_datum_field(datum, field, value.clone())?;
            }
        }
    }
    if !is_atom && !is_client && !is_image {
        return Ok(());
    }
    // These names exist without source declarations in BYOND's built-in atom
    // hierarchy.  Materialize only absent values so a project declaration on a
    // descendant retains its normal inherited/default-layer precedence.
    let atom_defaults: &[(&str, Value)] = &[
        ("alpha", Value::number(255.0)),
        ("appearance", Value::Null),
        ("appearance_flags", Value::number(0.0)),
        ("blend_mode", Value::number(0.0)),
        ("color", Value::Null),
        ("density", Value::number(0.0)),
        ("desc", Value::Null),
        ("dir", Value::number(2.0)),
        ("gender", Value::text("neuter")),
        ("icon", Value::Null),
        ("icon_state", Value::Null),
        ("invisibility", Value::number(0.0)),
        ("layer", Value::number(1.0)),
        ("loc", Value::Null),
        ("luminosity", Value::number(0.0)),
        ("maptext", Value::Null),
        ("maptext_height", Value::number(32.0)),
        ("maptext_width", Value::number(32.0)),
        ("maptext_x", Value::number(0.0)),
        ("maptext_y", Value::number(0.0)),
        ("mouse_opacity", Value::number(1.0)),
        ("mouse_over_pointer", Value::Null),
        ("name", Value::Null),
        ("opacity", Value::number(0.0)),
        ("particles", Value::Null),
        ("plane", Value::number(0.0)),
        ("pixel_x", Value::number(0.0)),
        ("pixel_y", Value::number(0.0)),
        ("pixel_w", Value::number(0.0)),
        ("pixel_z", Value::number(0.0)),
        ("render_source", Value::Null),
        ("render_target", Value::Null),
        ("suffix", Value::Null),
        ("transform", Value::Null),
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
        ("screen_loc", Value::Null),
        ("step_x", Value::number(0.0)),
        ("step_y", Value::number(0.0)),
        ("step_size", Value::number(32.0)),
    ];
    let mob_defaults = builtin_mob_defaults();
    let client_defaults = builtin_client_defaults();
    let image_defaults: &[(&str, Value)] = &[
        ("alpha", Value::number(255.0)),
        ("appearance", Value::Null),
        ("appearance_flags", Value::number(0.0)),
        ("blend_mode", Value::number(0.0)),
        ("color", Value::Null),
        ("dir", Value::number(2.0)),
        ("icon", Value::Null),
        ("icon_state", Value::Null),
        ("layer", Value::number(0.0)),
        ("loc", Value::Null),
        ("name", Value::Null),
        ("plane", Value::number(0.0)),
        ("transform", Value::Null),
    ];
    for (name, value) in is_atom
        .then_some(atom_defaults)
        .into_iter()
        .flatten()
        .chain(is_movable.then_some(movable_defaults).into_iter().flatten())
        .chain(
            is_mob
                .then_some(mob_defaults.as_slice())
                .into_iter()
                .flatten(),
        )
        .chain(
            is_client
                .then_some(client_defaults.as_slice())
                .into_iter()
                .flatten(),
        )
        .chain(is_image.then_some(image_defaults).into_iter().flatten())
    {
        let name = FieldName::parse(name).expect("built-in atom field name is valid");
        if heap.datum_field(datum, &name).is_err() {
            heap.set_datum_field(datum, name, value.clone())?;
        }
    }
    // Spatial contents owns live membership and therefore exists immediately.
    // BYOND/OpenDream lazily create an atom's appearance, verb, and visibility
    // collection wrappers on first access; allocating those six empty lists for
    // every map atom creates millions of permanently rooted heap identities.
    let list_fields: &[&str] = if is_client {
        &["images", "screen", "verbs"]
    } else if is_image {
        &["overlays", "underlays", "vis_contents"]
    } else {
        &["contents"]
    };
    for name in list_fields {
        let field = FieldName::parse(name).expect("built-in atom list field is valid");
        if heap.datum_field(datum, &field).is_err() {
            let list = heap.allocate_list();
            heap.set_datum_field(datum, field, Value::List(list))?;
        }
    }
    Ok(())
}

fn particle_defaults() -> &'static [(&'static str, Value)] {
    static DEFAULTS: std::sync::OnceLock<Vec<(&str, Value)>> = std::sync::OnceLock::new();
    DEFAULTS.get_or_init(|| {
        [
            "color",
            "width",
            "height",
            "count",
            "spawning",
            "bound1",
            "bound2",
            "gravity",
            "gradient",
            "transform",
            "icon",
            "icon_state",
            "lifespan",
            "fadein",
            "fade",
            "position",
            "velocity",
            "scale",
            "grow",
            "rotation",
            "spin",
            "friction",
            "drift",
        ]
        .into_iter()
        .map(|name| (name, Value::Null))
        .collect()
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WorldClockValues {
    realtime_deciseconds: f32,
    timeofday_deciseconds: f32,
    timezone_hours: f32,
}

fn world_clock_values(
    unix_millis: i64,
    local_hour: u16,
    local_minute: u16,
    local_second: u16,
    local_millis: u16,
    utc_offset_seconds: i64,
) -> WorldClockValues {
    const BYOND_EPOCH_UNIX_MILLIS: i64 = 946_684_800_000;
    let realtime_deciseconds = (unix_millis - BYOND_EPOCH_UNIX_MILLIS).div_euclid(100);
    let timeofday_deciseconds = i64::from(local_hour) * 36_000
        + i64::from(local_minute) * 600
        + i64::from(local_second) * 10
        + i64::from(local_millis) / 100;
    WorldClockValues {
        realtime_deciseconds: realtime_deciseconds as f32,
        timeofday_deciseconds: timeofday_deciseconds as f32,
        timezone_hours: utc_offset_seconds as f32 / 3_600.0,
    }
}

fn host_world_clock_values() -> WorldClockValues {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let unix_millis = i64::try_from(duration.as_millis()).unwrap_or(i64::MAX);
    let day_millis = unix_millis.rem_euclid(86_400_000);
    // Rust's standard library deliberately exposes no host civil-time API. A
    // headless Dream64 world therefore uses UTC as its coherent local zone
    // unless a future host adapter supplies a civil clock. This keeps
    // `timeofday` and `timezone` mutually correct without shelling out or
    // introducing a platform-specific unsafe dependency.
    world_clock_values(
        unix_millis,
        u16::try_from(day_millis / 3_600_000).unwrap_or_default(),
        u16::try_from(day_millis / 60_000 % 60).unwrap_or_default(),
        u16::try_from(day_millis / 1_000 % 60).unwrap_or_default(),
        u16::try_from(day_millis % 1_000).unwrap_or_default(),
        0,
    )
}

fn materialize_builtin_world_defaults(
    heap: &mut ValueHeap,
    datum: DatumId,
    type_path: &TypePath,
    world_name: &str,
) -> Result<(), ValueError> {
    let clock = host_world_clock_values();
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
        ("timezone", Value::number(clock.timezone_hours)),
        ("cpu", Value::number(0.0)),
        ("time", Value::number(0.0)),
        ("timeofday", Value::number(clock.timeofday_deciseconds)),
        ("realtime", Value::number(clock.realtime_deciseconds)),
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
        "/datum" => {
            insert("datum_flags", Value::number(0.0));
            insert("tag", Value::Null);
        }
        "/atom" => {
            for (name, value) in [
                ("alpha", Value::number(255.0)),
                ("appearance", Value::Null),
                ("appearance_flags", Value::number(0.0)),
                ("blend_mode", Value::number(0.0)),
                ("color", Value::Null),
                ("contents", Value::Null),
                ("density", Value::number(0.0)),
                ("desc", Value::Null),
                ("dir", Value::number(2.0)),
                ("gender", Value::text("neuter")),
                ("icon", Value::Null),
                ("icon_state", Value::Null),
                ("invisibility", Value::number(0.0)),
                ("layer", Value::number(1.0)),
                ("loc", Value::Null),
                ("luminosity", Value::number(0.0)),
                ("maptext", Value::Null),
                ("maptext_height", Value::number(32.0)),
                ("maptext_width", Value::number(32.0)),
                ("maptext_x", Value::number(0.0)),
                ("maptext_y", Value::number(0.0)),
                ("mouse_opacity", Value::number(1.0)),
                ("mouse_over_pointer", Value::Null),
                ("name", Value::Null),
                ("opacity", Value::number(0.0)),
                ("particles", Value::Null),
                ("filters", Value::Null),
                ("overlays", Value::Null),
                ("plane", Value::number(0.0)),
                ("pixel_x", Value::number(0.0)),
                ("pixel_y", Value::number(0.0)),
                ("pixel_w", Value::number(0.0)),
                ("pixel_z", Value::number(0.0)),
                ("render_source", Value::Null),
                ("render_target", Value::Null),
                ("suffix", Value::Null),
                ("transform", Value::Null),
                ("underlays", Value::Null),
                ("vis_contents", Value::Null),
                ("vis_locs", Value::Null),
                ("vis_flags", Value::number(0.0)),
                ("verbs", Value::Null),
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
                ("screen_loc", Value::Null),
                ("step_x", Value::number(0.0)),
                ("step_y", Value::number(0.0)),
                ("step_size", Value::number(32.0)),
            ] {
                insert(name, value);
            }
        }
        "/mob" => {
            for (name, value) in builtin_mob_defaults() {
                insert(name, value);
            }
        }
        "/client" => {
            for (name, value) in builtin_client_defaults() {
                insert(name, value);
            }
        }
        "/particles" => {
            for (name, value) in particle_defaults() {
                insert(name, value.clone());
            }
        }
        "/image" => {
            for (name, value) in [
                ("alpha", Value::number(255.0)),
                ("appearance", Value::Null),
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
    /// Unique retained instance initializers entered in the owner catalog.
    pub instance_initializer_candidates_indexed: usize,
    /// Unique instance-initializer programs compiled across every type plan.
    pub instance_initializer_unique_programs_compiled: usize,
    /// Shared compiled-program references installed into per-type plans.
    pub instance_initializer_plan_references: usize,
    /// Identifier/index probes used to construct initializer binding maps.
    pub initializer_binding_index_lookups: usize,
    /// Referenced bindings emitted across compiled initializer programs.
    pub initializer_bindings_emitted: usize,
    /// Immutable type metadata snapshots built for execution-state transfers.
    pub execution_metadata_builds: usize,
    /// Per-type inherited-default allocation plans built on first allocation.
    pub datum_allocation_plans_built: usize,
    /// Datums allocated inside a caller-owned persistent execution state.
    pub stateful_datums_allocated: usize,
    /// Project procedure bodies linked symbolically for initializer dispatch.
    pub initializer_module_deferred_procedures: usize,
    /// Deferred project bodies actually lowered while running initializers.
    pub initializer_module_materialized_procedures: usize,
    /// Distinct procedure selectors conservatively found in runtime
    /// initializer expressions.
    pub initializer_frontier_selectors: usize,
    /// Procedure specifications retained in the initializer module.
    pub initializer_module_procedures: usize,
    /// Whether an indirect `call()` required the complete project inventory.
    pub initializer_complete_symbol_inventory: usize,
    /// Exact construction type paths retained by initializer analysis.
    pub initializer_typed_constructor_targets: usize,
    /// Whether at least one initializer required the full dynamic `New` set.
    pub initializer_dynamic_constructor_frontier: usize,
    /// Existing runtime slots updated through the canonical path index.
    pub indexed_runtime_variable_updates: usize,
    /// Runtime slots read through the canonical path index while synchronizing
    /// initializer state.
    pub indexed_runtime_variable_reads: usize,
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

/// Coarse construction phases exposed for cold-boot diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeImageConstructionPhase {
    /// Inventory global/static/instance variables and initialization plans.
    VariableRegistry,
    /// Build runtime type/default and reflection metadata.
    TypeInventory,
    /// Materialize compile-time instance constants.
    InstanceConstants,
    /// Build the inherited initial-value snapshot used by VM execution.
    ExecutionMetadata,
    /// Build semantic procedure identities and dependency indices.
    ProcedureRegistry,
    /// Resolve the initializer frontier and construct its symbolic VM module.
    InitializerModuleLink,
    /// Compile the unique runtime instance-initializer entry points.
    InstanceInitializerCompilation,
    /// Execute source-ordered global and type-static initializers.
    GlobalInitializerExecution,
    /// Capture final counters and immutable runtime state.
    Finalization,
}

impl RuntimeImageConstructionPhase {
    /// Stable machine-readable phase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VariableRegistry => "variable-registry",
            Self::TypeInventory => "type-inventory",
            Self::InstanceConstants => "instance-constants",
            Self::ExecutionMetadata => "execution-metadata",
            Self::ProcedureRegistry => "procedure-registry",
            Self::InitializerModuleLink => "initializer-module-link",
            Self::InstanceInitializerCompilation => "instance-initializer-compilation",
            Self::GlobalInitializerExecution => "global-initializer-execution",
            Self::Finalization => "finalization",
        }
    }
}

/// Boundary event emitted while constructing a [`RuntimeImage`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeImageConstructionEvent {
    /// Phase entering or completing.
    pub phase: RuntimeImageConstructionPhase,
    /// Whether this event marks phase completion rather than entry.
    pub completed: bool,
    /// Wall-clock duration for a completed phase; zero on entry.
    pub elapsed: Duration,
    /// Deterministic number of principal items handled, when available.
    pub items: Option<usize>,
}

fn begin_runtime_phase(
    observer: &mut impl FnMut(RuntimeImageConstructionEvent),
    phase: RuntimeImageConstructionPhase,
) -> Instant {
    observer(RuntimeImageConstructionEvent {
        phase,
        completed: false,
        elapsed: Duration::ZERO,
        items: None,
    });
    Instant::now()
}

fn complete_runtime_phase(
    observer: &mut impl FnMut(RuntimeImageConstructionEvent),
    phase: RuntimeImageConstructionPhase,
    started: Instant,
    items: Option<usize>,
) {
    observer(RuntimeImageConstructionEvent {
        phase,
        completed: true,
        elapsed: started.elapsed(),
        items,
    });
}

/// A deterministic runtime-ready constant image for one compiled project.
pub struct RuntimeImage {
    heap: ValueHeap,
    variables: Vec<RuntimeVariable>,
    types: BTreeMap<TypePath, RuntimeType>,
    type_paths: Arc<BTreeSet<TypePath>>,
    type_parents: Arc<BTreeMap<TypePath, Option<TypePath>>>,
    initial_values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,
    shared_fields: Arc<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>>,
    global_types: BTreeMap<String, TypePath>,
    world_name: String,
    canonical_world: Option<DatumId>,
    binding_index: RuntimeBindingIndex,
    runtime_variable_indices: BTreeMap<String, usize>,
    global_variable_indices: BTreeMap<FieldName, usize>,
    diagnostics: Vec<RuntimeInitializerDiagnostic>,
    instance_initializers: Vec<InstanceInitializerCandidate>,
    instance_initializer_indices_by_owner: BTreeMap<TypePath, Vec<usize>>,
    compiled_instance_initializers: BTreeMap<usize, CompiledInstanceInitializer>,
    vm_instance_initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
    vm_instance_initializer_module: Option<Arc<Module>>,
    instance_initializer_plans: BTreeMap<TypePath, Arc<[CompiledInstanceInitializer]>>,
    datum_allocation_plans: BTreeMap<TypePath, DatumAllocationPlan>,
    procedure_static_locals: BTreeMap<(String, u16), Value>,
    project_root: PathBuf,
    stats: RuntimeImageStats,
}

/// Counts of rebuildable allocation caches released after bulk preflight and
/// world materialization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeCacheReleaseStats {
    /// Per-type inherited initializer plans.
    pub initializer_plans: usize,
    /// Standalone initializer programs referenced by those plans.
    pub initializer_programs: usize,
    /// Per-type inherited default/allocation plans.
    pub allocation_plans: usize,
}

#[derive(Clone)]
struct CompiledInstanceInitializer {
    path: String,
    field: FieldName,
    action: CompiledInstanceInitializerAction,
}

#[derive(Clone)]
enum CompiledInstanceInitializerAction {
    Constant(Value),
    Program(Arc<InitializerProgram>),
}

#[derive(Clone)]
struct InstanceInitializerCandidate {
    entry: VariableEntry,
    path: String,
    constant: Option<Value>,
}

#[derive(Clone)]
struct DatumAllocationPlan {
    defaults: Arc<[DatumDefaults]>,
    ancestors: Arc<[TypePath]>,
    is_atom: bool,
    is_movable: bool,
    is_mob: bool,
    is_client: bool,
    is_particles: bool,
    is_image: bool,
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
            } else if entry.storage == StorageClass::Global && entry.owner.is_none() {
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

fn inherited_runtime_initializer_precedes(
    types: &BTreeMap<TypePath, RuntimeType>,
    runtime_fields: &BTreeMap<TypePath, BTreeMap<FieldName, Vec<usize>>>,
    owner: &TypePath,
    ordinal: usize,
) -> bool {
    if runtime_fields.get(owner).is_some_and(|fields| {
        fields
            .values()
            .flatten()
            .any(|candidate| *candidate < ordinal)
    }) {
        return true;
    }

    let mut current = types
        .get(owner)
        .and_then(|runtime_type| runtime_type.parent.clone());
    let mut visited = BTreeSet::new();
    while let Some(path) = current {
        if !visited.insert(path.clone()) {
            break;
        }
        if runtime_fields
            .get(&path)
            .is_some_and(|fields| !fields.is_empty())
        {
            return true;
        }
        current = types
            .get(&path)
            .and_then(|runtime_type| runtime_type.parent.clone());
    }
    false
}

fn type_parent_metadata(
    types: &BTreeMap<TypePath, RuntimeType>,
) -> BTreeMap<TypePath, Option<TypePath>> {
    types
        .iter()
        .map(|(path, runtime_type)| (path.clone(), runtime_type.parent.clone()))
        .collect()
}

fn shared_reflection_fields(
    registry: &VariableRegistry,
    parents: &BTreeMap<TypePath, Option<TypePath>>,
) -> Result<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>, RuntimeImageError> {
    let mut direct = BTreeMap::<TypePath, BTreeMap<FieldName, FieldName>>::new();
    for entry in registry
        .entries()
        .iter()
        .filter(|entry| entry.storage != StorageClass::Instance && entry.owner.is_some())
    {
        let owner = parse_type_path(&entry.owner.as_ref().expect("filtered owner").path)?;
        let name = variable_field(&entry.path)?;
        direct
            .entry(owner)
            .or_default()
            .insert(name, FieldName::static_storage(&entry.path));
    }
    let mut result = BTreeMap::new();
    for path in parents.keys() {
        let mut hierarchy = Vec::new();
        let mut current = Some(path.clone());
        while let Some(candidate) = current {
            hierarchy.push(candidate.clone());
            current = parents.get(&candidate).cloned().flatten();
        }
        hierarchy.reverse();
        let mut fields = BTreeMap::new();
        for ancestor in hierarchy {
            if let Some(entries) = direct.get(&ancestor) {
                fields.extend(entries.clone());
            }
        }
        result.insert(path.clone(), fields);
    }
    Ok(result)
}

fn declared_global_types(
    compilation: &Compilation,
    registry: &VariableRegistry,
) -> BTreeMap<String, TypePath> {
    let mut result = BTreeMap::new();
    for entry in registry
        .entries()
        .iter()
        .filter(|entry| entry.storage == StorageClass::Global && entry.owner.is_none())
    {
        let Some(name) = entry.path.rsplit('/').next() else {
            continue;
        };
        if let Some(path) = declared_variable_type(compilation, entry) {
            result.insert(name.to_owned(), path);
        }
    }
    result
}

fn declared_variable_type(compilation: &Compilation, entry: &VariableEntry) -> Option<TypePath> {
    let name = entry.path.rsplit('/').next()?;
    let definition = compilation
        .syntax(entry.file_id)?
        .definitions
        .get(entry.definition_index)?;
    let identifiers = definition
        .header
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let name_index = identifiers
        .iter()
        .rposition(|candidate| *candidate == name)?;
    let var_index = identifiers
        .iter()
        .position(|candidate| *candidate == "var")?;
    let segments = identifiers[var_index + 1..name_index]
        .iter()
        .filter(|segment| !["global", "static", "const", "tmp", "final"].contains(segment))
        .copied()
        .collect::<Vec<_>>();
    (!segments.is_empty())
        .then(|| TypePath::parse(&format!("/{}", segments.join("/"))).ok())
        .flatten()
}

#[derive(Default)]
struct InitializerProcedureFrontier {
    selectors: BTreeSet<String>,
    constructed_types: BTreeSet<TypePath>,
    requires_dynamic_constructors: bool,
    requires_complete_inventory: bool,
}

impl InitializerProcedureFrontier {
    fn include(&mut self, compilation: &Compilation, entry: &VariableEntry) {
        let Some(initializer) = &entry.initializer else {
            return;
        };
        let inferred_type = declared_variable_type(compilation, entry);
        let mut index = 0usize;
        while index < initializer.tokens.len() {
            let token = &initializer.tokens[index];
            match &token.kind {
                TokenKind::Identifier(name) => {
                    let is_call_head = matches!(
                        initializer.tokens.get(index + 1).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    );
                    if name == "call" && is_call_head {
                        // `call(receiver, selector)` may obtain its selector
                        // from arbitrary runtime data. A reduced inventory is
                        // not sound for that expression.
                        self.requires_complete_inventory = true;
                    } else if name == "new" {
                        let mut cursor = index + 1;
                        if matches!(
                            initializer.tokens.get(cursor).map(|token| &token.kind),
                            Some(TokenKind::Operator(operator)) if operator == "/"
                        ) {
                            let mut segments = Vec::new();
                            while cursor + 1 < initializer.tokens.len()
                                && matches!(
                                    &initializer.tokens[cursor].kind,
                                    TokenKind::Operator(operator) if operator == "/"
                                )
                            {
                                let TokenKind::Identifier(segment) =
                                    &initializer.tokens[cursor + 1].kind
                                else {
                                    break;
                                };
                                segments.push(segment.clone());
                                cursor += 2;
                            }
                            if !segments.is_empty()
                                && (segments.len() != 1 || segments[0] != "list")
                            {
                                if let Ok(path) =
                                    TypePath::parse(&format!("/{}", segments.join("/")))
                                {
                                    self.constructed_types.insert(path);
                                } else {
                                    self.requires_dynamic_constructors = true;
                                }
                            }
                            index = cursor.saturating_sub(1);
                        } else if matches!(
                            initializer.tokens.get(cursor).map(|token| &token.kind),
                            None | Some(TokenKind::Punctuation('('))
                        ) {
                            if let Some(path) = &inferred_type {
                                if path.as_str() != "/list" {
                                    self.constructed_types.insert(path.clone());
                                }
                            } else {
                                self.requires_dynamic_constructors = true;
                            }
                        } else {
                            // `new type_expression(...)` is selected at runtime.
                            self.requires_dynamic_constructors = true;
                            if let Some(open) = initializer.tokens[cursor..]
                                .iter()
                                .position(|token| matches!(token.kind, TokenKind::Punctuation('(')))
                            {
                                // Do not reinterpret the dynamic type operand
                                // as a call selector. Arguments after the open
                                // parenthesis remain visible to this scan.
                                index = cursor + open;
                            }
                        }
                    } else if is_call_head {
                        self.selectors.insert(name.clone());
                    }
                }
                TokenKind::String(text)
                | TokenKind::RawString(text)
                | TokenKind::TextBlock(text) => {
                    // Literal procedure paths remain first-class references,
                    // including paths passed through text2path(). Ordinary
                    // user-facing strings are not procedure selectors.
                    if let Some(selector) = proc_selector_from_text_path(text) {
                        self.selectors.insert(selector.to_owned());
                    }
                }
                TokenKind::Operator(operator) if operator == "/" => {
                    let mut cursor = index;
                    let mut segments = Vec::new();
                    while cursor + 1 < initializer.tokens.len()
                        && matches!(&initializer.tokens[cursor].kind, TokenKind::Operator(operator) if operator == "/")
                    {
                        let TokenKind::Identifier(segment) = &initializer.tokens[cursor + 1].kind
                        else {
                            break;
                        };
                        segments.push(segment.as_str());
                        cursor += 2;
                    }
                    if let Some(proc_index) = segments.iter().position(|segment| *segment == "proc")
                        && let Some(selector) = segments.get(proc_index + 1)
                    {
                        self.selectors.insert((*selector).to_owned());
                        index = cursor.saturating_sub(1);
                    }
                }
                _ => {}
            }
            index += 1;
        }
    }
}

fn proc_selector_from_text_path(text: &str) -> Option<&str> {
    let segments = text.strip_prefix('/')?.split('/').collect::<Vec<_>>();
    let proc_index = segments.iter().position(|segment| *segment == "proc")?;
    segments
        .get(proc_index + 1)
        .copied()
        .filter(|selector| !selector.is_empty())
}

#[derive(Default)]
struct InitializerBindingReferences {
    bare: BTreeSet<String>,
    qualified: BTreeSet<(String, String)>,
}

fn initializer_binding_references(
    tokens: &[dm_lexer::SpannedToken],
) -> InitializerBindingReferences {
    let mut references = InitializerBindingReferences::default();
    for token in tokens {
        if let TokenKind::Identifier(name) = &token.kind {
            references.bare.insert(name.clone());
        }
    }
    for window in tokens.windows(3) {
        let [receiver, operator, member] = window else {
            continue;
        };
        let TokenKind::Identifier(receiver) = &receiver.kind else {
            continue;
        };
        if !matches!(&operator.kind, TokenKind::Operator(operator) if matches!(operator.as_str(), "." | "?." | "::"))
        {
            continue;
        }
        let TokenKind::Identifier(member) = &member.kind else {
            continue;
        };
        references
            .qualified
            .insert((receiver.clone(), member.clone()));
    }
    references
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
        Self::from_compilation_with_observer(compilation, |_| {})
    }

    /// Materializes one frontend snapshot while reporting structured phase
    /// boundaries to `observer`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::from_compilation`]. A phase that
    /// fails emits its entry event but no completion event.
    pub fn from_compilation_with_observer(
        compilation: &Compilation,
        mut observer: impl FnMut(RuntimeImageConstructionEvent),
    ) -> Result<Self, RuntimeImageError> {
        let phase = RuntimeImageConstructionPhase::VariableRegistry;
        let phase_started = begin_runtime_phase(&mut observer, phase);
        let registry = VariableRegistry::build(compilation);
        let plans = registry.initialization_plans();
        let binding_index = RuntimeBindingIndex::build(&registry)?;
        complete_runtime_phase(
            &mut observer,
            phase,
            phase_started,
            Some(registry.entries().len()),
        );

        let phase = RuntimeImageConstructionPhase::TypeInventory;
        let phase_started = begin_runtime_phase(&mut observer, phase);
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
        // Reflection of shared fields needs only the parent graph here. The
        // much larger inherited-initial-value snapshot is built once after
        // all compile-time instance constants have been applied.
        let type_parents = type_parent_metadata(&types);
        let shared_fields = shared_reflection_fields(&registry, &type_parents)?;
        let global_types = declared_global_types(compilation, &registry);
        complete_runtime_phase(&mut observer, phase, phase_started, Some(types.len()));
        let mut image = Self {
            heap: ValueHeap::new(),
            variables: Vec::new(),
            types,
            type_paths,
            type_parents: Arc::new(type_parents),
            initial_values: Arc::new(BTreeMap::new()),
            shared_fields: Arc::new(shared_fields),
            global_types,
            world_name,
            canonical_world: None,
            binding_index,
            runtime_variable_indices: BTreeMap::new(),
            global_variable_indices: BTreeMap::new(),
            diagnostics: Vec::new(),
            instance_initializers: Vec::new(),
            instance_initializer_indices_by_owner: BTreeMap::new(),
            compiled_instance_initializers: BTreeMap::new(),
            vm_instance_initializers: Arc::new(BTreeMap::new()),
            vm_instance_initializer_module: None,
            instance_initializer_plans: BTreeMap::new(),
            datum_allocation_plans: BTreeMap::new(),
            procedure_static_locals: BTreeMap::new(),
            project_root: compilation.project().root_directory.clone(),
            stats: RuntimeImageStats {
                variables: registry.entries().len(),
                initializer_steps: plans.global_steps.len()
                    + plans
                        .type_defaults
                        .iter()
                        .map(|plan| plan.steps.len())
                        .sum::<usize>(),
                ..RuntimeImageStats::default()
            },
        };

        let phase = RuntimeImageConstructionPhase::InstanceConstants;
        let phase_started = begin_runtime_phase(&mut observer, phase);

        // Shared declarations exist even without an explicit initializer.
        // Type-owned `var/global` is BYOND static storage, not a project bare
        // global, and therefore uses the same owner-qualified slot as
        // `var/static`.
        for entry in registry.entries().iter().filter(|entry| {
            entry.storage != StorageClass::Instance
                && entry.assignment == dm_globals::AssignmentKind::Declaration
                && entry.initializer.is_none()
        }) {
            let field = if entry.storage == StorageClass::Global && entry.owner.is_none() {
                variable_field(&entry.path)?
            } else {
                FieldName::static_storage(&entry.path)
            };
            image.variables.push(RuntimeVariable {
                path: entry.path.clone(),
                storage: entry.storage,
                value: Value::Null,
                ordinal: entry.ordinal,
            });
            image
                .runtime_variable_indices
                .entry(entry.path.clone())
                .or_insert(image.variables.len() - 1);
            image
                .global_variable_indices
                .insert(field, image.variables.len() - 1);
        }

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
        let instance_step_count = instance_steps.len();
        let mut runtime_initializer_fields =
            BTreeMap::<TypePath, BTreeMap<FieldName, Vec<usize>>>::new();
        for step in &instance_steps {
            if !matches!(
                step.evaluation,
                ConstantEvaluation::Value(ConstantValue::List(_))
                    | ConstantEvaluation::Unsupported(_)
            ) {
                continue;
            }
            let entry = &registry.entries()[step.entry_index];
            let owner = entry
                .owner
                .as_ref()
                .ok_or_else(|| RuntimeImageError::MissingOwner(step.path.clone()))?;
            runtime_initializer_fields
                .entry(parse_type_path(&owner.path)?)
                .or_default()
                .entry(variable_field(&step.path)?)
                .or_default()
                .push(step.ordinal);
        }
        let mut state = image.take_execution_state();
        for step in instance_steps {
            let entry = &registry.entries()[step.entry_index];
            match &step.evaluation {
                ConstantEvaluation::Value(ConstantValue::List(_))
                | ConstantEvaluation::Unsupported(_) => {
                    let owner = entry
                        .owner
                        .as_ref()
                        .ok_or_else(|| RuntimeImageError::MissingOwner(step.path.clone()))?;
                    let owner = parse_type_path(&owner.path)?;
                    let index = image.instance_initializers.len();
                    image
                        .instance_initializers
                        .push(InstanceInitializerCandidate {
                            entry: entry.clone(),
                            path: step.path.clone(),
                            constant: None,
                        });
                    image
                        .instance_initializer_indices_by_owner
                        .entry(owner)
                        .or_default()
                        .push(index);
                    image.stats.instance_initializer_candidates_indexed += 1;
                }
                ConstantEvaluation::Value(constant) => {
                    let value = image.convert_constant_in(constant, state.heap_mut())?;
                    let owner = entry
                        .owner
                        .as_ref()
                        .ok_or_else(|| RuntimeImageError::MissingOwner(step.path.clone()))?;
                    let owner = parse_type_path(&owner.path)?;
                    let replay = inherited_runtime_initializer_precedes(
                        &image.types,
                        &runtime_initializer_fields,
                        &owner,
                        step.ordinal,
                    );
                    image.apply_step_value(entry, step, value.clone())?;
                    if replay {
                        let index = image.instance_initializers.len();
                        image
                            .instance_initializers
                            .push(InstanceInitializerCandidate {
                                entry: entry.clone(),
                                path: step.path.clone(),
                                constant: Some(value),
                            });
                        image
                            .instance_initializer_indices_by_owner
                            .entry(owner)
                            .or_default()
                            .push(index);
                        image.stats.instance_initializer_candidates_indexed += 1;
                    }
                    image.stats.constants_materialized += 1;
                }
            }
        }
        // A subtype may explicitly redeclare an inherited runtime-initialized
        // field without `=`, which is a real null override in BYOND rather than
        // an absent assignment. Retain that null at its source position when
        // an earlier same-field runtime initializer could overwrite it.
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
            if !inherited_runtime_initializer_precedes(
                &image.types,
                &runtime_initializer_fields,
                &owner,
                entry.ordinal,
            ) {
                continue;
            }
            let index = image.instance_initializers.len();
            image
                .instance_initializers
                .push(InstanceInitializerCandidate {
                    entry: entry.clone(),
                    path: entry.path.clone(),
                    constant: Some(Value::Null),
                });
            image
                .instance_initializer_indices_by_owner
                .entry(owner)
                .or_default()
                .push(index);
            image.stats.instance_initializer_candidates_indexed += 1;
        }
        let candidates = &image.instance_initializers;
        for indices in image.instance_initializer_indices_by_owner.values_mut() {
            indices.sort_by_key(|index| candidates[*index].entry.ordinal);
        }
        image.restore_execution_state(state);
        complete_runtime_phase(
            &mut observer,
            phase,
            phase_started,
            Some(instance_step_count),
        );
        // This is the sole whole-tree inherited-default snapshot. Neither
        // world allocation nor global/static initialization mutates type
        // defaults, so rebuilding it later only repeats O(types * inherited
        // fields) cloning without changing execution semantics.
        let phase = RuntimeImageConstructionPhase::ExecutionMetadata;
        let phase_started = begin_runtime_phase(&mut observer, phase);
        image.refresh_execution_metadata();
        complete_runtime_phase(&mut observer, phase, phase_started, Some(image.types.len()));

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
        let mut initializer_frontier = InitializerProcedureFrontier::default();
        for candidate in &image.instance_initializers {
            if candidate.constant.is_none() {
                initializer_frontier.include(compilation, &candidate.entry);
            }
        }
        for step in &plans.global_steps {
            if matches!(step.evaluation, ConstantEvaluation::Unsupported(_)) {
                initializer_frontier.include(compilation, &registry.entries()[step.entry_index]);
            }
        }
        image.stats.initializer_frontier_selectors = initializer_frontier.selectors.len();
        image.stats.initializer_typed_constructor_targets =
            initializer_frontier.constructed_types.len();
        image.stats.initializer_dynamic_constructor_frontier =
            usize::from(initializer_frontier.requires_dynamic_constructors);
        image.stats.initializer_complete_symbol_inventory =
            usize::from(initializer_frontier.requires_complete_inventory);
        let registry_phase = RuntimeImageConstructionPhase::ProcedureRegistry;
        let registry_started = begin_runtime_phase(&mut observer, registry_phase);
        let procedures = ProcedureRegistry::build(compilation);
        complete_runtime_phase(
            &mut observer,
            registry_phase,
            registry_started,
            Some(procedures.procedures().len()),
        );
        let phase = RuntimeImageConstructionPhase::InitializerModuleLink;
        let phase_started = begin_runtime_phase(&mut observer, phase);
        let initializer_executable = if initializer_frontier.requires_complete_inventory {
            procedures.compile_vm_all_symbolic_deferred(compilation)
        } else {
            procedures.compile_vm_initializer_typed_frontier_symbolic_deferred(
                compilation,
                initializer_frontier.selectors.iter().map(String::as_str),
                initializer_frontier.constructed_types.iter(),
                initializer_frontier.requires_dynamic_constructors,
            )
        };
        if let Ok(executable) = &initializer_executable {
            image.stats.initializer_module_procedures = executable.stats().procedures;
        }
        let mut initializer_module = initializer_executable
            .ok()
            .map(|executable| executable.module().clone());
        complete_runtime_phase(
            &mut observer,
            phase,
            phase_started,
            Some(image.stats.initializer_module_procedures),
        );

        let phase = RuntimeImageConstructionPhase::InstanceInitializerCompilation;
        let phase_started = begin_runtime_phase(&mut observer, phase);
        if let Some(module) = initializer_module.as_mut() {
            image.vm_instance_initializers =
                Arc::new(image.compile_vm_instance_initializers(module)?);
            image.vm_instance_initializer_module = Some(Arc::new(module.clone()));
            state.set_instance_initializers(
                Arc::clone(&image.vm_instance_initializers),
                image.vm_instance_initializer_module.clone(),
            );
        }
        complete_runtime_phase(
            &mut observer,
            phase,
            phase_started,
            Some(image.vm_instance_initializers.values().map(Vec::len).sum()),
        );

        let phase = RuntimeImageConstructionPhase::GlobalInitializerExecution;
        let phase_started = begin_runtime_phase(&mut observer, phase);
        let mut global_steps = plans.global_steps.iter().collect::<Vec<_>>();
        global_steps.sort_by_key(|step| step.ordinal);
        let global_step_count = global_steps.len();
        for step in global_steps {
            let entry = &registry.entries()[step.entry_index];
            match &step.evaluation {
                ConstantEvaluation::Value(constant) => {
                    let value = image.convert_constant_in(constant, state.heap_mut())?;
                    image.apply_step_value(entry, step, value)?;
                    image.sync_initializer_global(entry, step, &mut state)?;
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
                            image.sync_initializer_global(entry, step, &mut state)?;
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
        if let Some(module) = initializer_module.as_ref() {
            image.stats.initializer_module_deferred_procedures = module.deferred_procedure_count();
            image.stats.initializer_module_materialized_procedures =
                module.materialized_deferred_procedure_count();
        }
        image.restore_execution_state(state);
        complete_runtime_phase(&mut observer, phase, phase_started, Some(global_step_count));

        let phase = RuntimeImageConstructionPhase::Finalization;
        let phase_started = begin_runtime_phase(&mut observer, phase);
        image.stats.runtime_variables = image.variables.len();
        image.stats.runtime_types = image.types.len();
        image.stats.default_layers = image
            .types
            .values()
            .filter(|runtime_type| runtime_type.defaults.fields().len() != 0)
            .count();
        image.stats.unsupported_initializers = image.diagnostics.len();
        complete_runtime_phase(
            &mut observer,
            phase,
            phase_started,
            Some(image.stats.runtime_variables),
        );
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

    /// Releases rebuildable caches populated by initializer preflight and bulk
    /// world allocation.
    ///
    /// The source initializer inventory and shared VM initializer module remain
    /// intact. A later [`Self::allocate_datum`] call therefore remains correct;
    /// it simply rebuilds the requested type's compact plans lazily. This is
    /// useful before a separate large compilation phase once bulk map
    /// allocation has completed.
    pub fn release_allocation_caches(&mut self) -> RuntimeCacheReleaseStats {
        let stats = RuntimeCacheReleaseStats {
            initializer_plans: self.instance_initializer_plans.len(),
            initializer_programs: self.compiled_instance_initializers.len(),
            allocation_plans: self.datum_allocation_plans.len(),
        };
        self.instance_initializer_plans.clear();
        self.compiled_instance_initializers.clear();
        self.datum_allocation_plans.clear();
        stats
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
        state.set_shared_fields(Arc::clone(&self.shared_fields));
        state.set_instance_initializers(
            Arc::clone(&self.vm_instance_initializers),
            self.vm_instance_initializer_module.clone(),
        );
        state.set_project_root(self.project_root.clone());
        state.set_procedure_static_locals(std::mem::take(&mut self.procedure_static_locals));
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
    pub fn restore_execution_state(&mut self, mut state: ExecutionState) {
        for (field, index) in &self.global_variable_indices {
            if let Some(value) = state.global(field) {
                self.variables[*index].value.clone_from(value);
            }
        }
        self.procedure_static_locals = state.take_procedure_static_locals();
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
            allocation.is_mob,
            allocation.is_client,
            allocation.is_particles,
            allocation.is_image,
        )?;
        materialize_builtin_world_defaults(&mut self.heap, datum, type_path, &self.world_name)?;
        let plan = self.instance_initializer_plan(type_path, &allocation.ancestors)?;
        if !plan.is_empty() {
            let mut state = self.take_execution_state();
            let result: Result<(), RuntimeImageError> = (|| {
                for initializer in plan.iter() {
                    let value = match &initializer.action {
                        CompiledInstanceInitializerAction::Constant(value) => value.clone(),
                        CompiledInstanceInitializerAction::Program(program) => {
                            execute_module_in_context(
                                program.module(),
                                program.entry(),
                                &[],
                                &mut state,
                                &ExecutionContext::new(Value::Datum(datum), Value::Null),
                            )
                            .map_err(|error| {
                                RuntimeImageError::InstanceInitializer {
                                    path: initializer.path.clone(),
                                    message: error.message,
                                }
                            })?
                        }
                    };
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
            allocation.is_mob,
            allocation.is_client,
            allocation.is_particles,
            allocation.is_image,
        )?;
        materialize_builtin_world_defaults(state.heap_mut(), datum, type_path, &self.world_name)?;
        let plan = self.instance_initializer_plan(type_path, &allocation.ancestors)?;
        for initializer in plan.iter() {
            let value = match &initializer.action {
                CompiledInstanceInitializerAction::Constant(value) => value.clone(),
                CompiledInstanceInitializerAction::Program(program) => execute_module_in_context(
                    program.module(),
                    program.entry(),
                    &[],
                    state,
                    &ExecutionContext::new(Value::Datum(datum), Value::Null),
                )
                .map_err(|error| RuntimeImageError::InstanceInitializer {
                    path: initializer.path.clone(),
                    message: error.message,
                })?,
            };
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
        let mut is_mob = false;
        let mut is_client = false;
        let mut is_particles = false;
        let mut is_image = false;
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
            is_mob |= path.as_str() == "/mob";
            is_client |= path.as_str() == "/client";
            is_particles |= path.as_str() == "/particles";
            is_image |= path.as_str() == "/image";
            chain.push(runtime_type.defaults.clone());
            current.clone_from(&runtime_type.parent);
        }
        chain.reverse();
        let ancestors = chain
            .iter()
            .map(|defaults| defaults.type_path().clone())
            .collect::<Vec<_>>();
        let plan = DatumAllocationPlan {
            defaults: Arc::from(chain),
            ancestors: Arc::from(ancestors),
            is_atom,
            is_movable,
            is_mob,
            is_client,
            is_particles,
            is_image,
        };
        self.datum_allocation_plans
            .insert(type_path.clone(), plan.clone());
        self.stats.datum_allocation_plans_built += 1;
        Ok(plan)
    }

    fn instance_initializer_plan(
        &mut self,
        type_path: &TypePath,
        ancestors: &[TypePath],
    ) -> Result<Arc<[CompiledInstanceInitializer]>, RuntimeImageError> {
        if let Some(plan) = self.instance_initializer_plans.get(type_path) {
            return Ok(plan.clone());
        }

        let applicable = ancestors
            .iter()
            .filter_map(|owner| self.instance_initializer_indices_by_owner.get(owner))
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let mut plan = Vec::with_capacity(applicable.len());
        for index in applicable {
            if let Some(initializer) = self.compiled_instance_initializers.get(&index) {
                plan.push(initializer.clone());
                continue;
            }
            let candidate = self.instance_initializers[index].clone();
            let entry = candidate.entry;
            let path = candidate.path;
            if let Some(value) = candidate.constant {
                let compiled = CompiledInstanceInitializer {
                    path: path.clone(),
                    field: variable_field(&path)?,
                    action: CompiledInstanceInitializerAction::Constant(value),
                };
                self.compiled_instance_initializers
                    .insert(index, compiled.clone());
                plan.push(compiled);
                continue;
            }
            let initializer = entry
                .initializer
                .as_ref()
                .ok_or_else(|| RuntimeImageError::MissingInitializer(path.clone()))?;
            let bindings = self.initializer_bindings(&entry).map_err(|failure| {
                RuntimeImageError::InstanceInitializer {
                    path: path.clone(),
                    message: failure.message,
                }
            })?;
            let program =
                compile_initializer(&initializer.tokens, &bindings, None).map_err(|error| {
                    RuntimeImageError::InstanceInitializer {
                        path: path.clone(),
                        message: error.message,
                    }
                })?;
            let compiled = CompiledInstanceInitializer {
                path: path.clone(),
                field: variable_field(&path)?,
                action: CompiledInstanceInitializerAction::Program(Arc::new(program)),
            };
            self.compiled_instance_initializers
                .insert(index, compiled.clone());
            self.stats.instance_initializer_unique_programs_compiled += 1;
            plan.push(compiled);
        }
        self.stats.instance_initializer_plan_references += plan.len();
        let plan = Arc::<[CompiledInstanceInitializer]>::from(plan);
        self.instance_initializer_plans
            .insert(type_path.clone(), Arc::clone(&plan));
        self.stats.instance_initializer_plans_compiled += 1;
        Ok(plan)
    }

    fn compile_vm_instance_initializers(
        &mut self,
        module: &mut Module,
    ) -> Result<BTreeMap<TypePath, Vec<InstanceInitializer>>, RuntimeImageError> {
        let mut catalog = BTreeMap::<TypePath, Vec<InstanceInitializer>>::new();
        let ordered = self
            .instance_initializer_indices_by_owner
            .iter()
            .flat_map(|(owner, indices)| indices.iter().map(|index| (owner.clone(), *index)))
            .collect::<Vec<_>>();
        for (catalog_owner, index) in ordered {
            let candidate = self.instance_initializers[index].clone();
            let entry = candidate.entry;
            let path = candidate.path;
            let owner = entry
                .owner
                .as_ref()
                .ok_or_else(|| RuntimeImageError::MissingOwner(entry.path.clone()))?;
            let owner = parse_type_path(&owner.path)?;
            debug_assert_eq!(owner, catalog_owner);
            let field = variable_field(&path)?;
            if let Some(value) = candidate.constant {
                catalog
                    .entry(owner)
                    .or_default()
                    .push(InstanceInitializer::Constant { field, value });
                continue;
            }
            let initializer = entry
                .initializer
                .as_ref()
                .ok_or_else(|| RuntimeImageError::MissingInitializer(path.clone()))?;
            let bindings = self.initializer_bindings(&entry).map_err(|failure| {
                RuntimeImageError::InstanceInitializer {
                    path: path.clone(),
                    message: failure.message,
                }
            })?;
            let Ok(program) =
                compile_initializer_into_module(&initializer.tokens, &bindings, module)
            else {
                // Preserve lazy preflight behavior for invalid/unreachable
                // type defaults; the existing per-type plan reports the
                // source-mapped error when that type is requested.
                continue;
            };
            catalog
                .entry(owner)
                .or_default()
                .push(InstanceInitializer::Program {
                    field,
                    entry: program,
                });
        }
        Ok(catalog)
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

    /// Evaluates a live-datum expression by appending its entry point to an
    /// existing executable module.
    ///
    /// Existing procedure identities and deferred bodies remain unchanged;
    /// this is intended for map overrides discovered only after world
    /// allocation, when rebuilding the lifecycle closure would be wasteful.
    /// The appended expression retains its expanded source spans and may call
    /// global project procedures already linked in `module`.
    pub fn evaluate_datum_expression_linked(
        &self,
        datum: DatumId,
        expression: &str,
        state: &mut ExecutionState,
        module: &mut Module,
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
        let entry =
            compile_initializer_into_module(&tokens, &bindings, module).map_err(|error| {
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
            module,
            entry,
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
            let value = normalize_engine_instance_value(&owner, &field, value);
            let runtime_type = self
                .types
                .get_mut(&owner)
                .ok_or_else(|| RuntimeImageError::UnknownType(owner.clone()))?;
            runtime_type.defaults.set(field, value);
            return Ok(());
        }
        if let Some(index) = self.runtime_variable_indices.get(&step.path).copied() {
            let variable = &mut self.variables[index];
            variable.value = value;
            variable.ordinal = step.ordinal;
            variable.storage = step.storage;
            self.stats.indexed_runtime_variable_updates += 1;
            return Ok(());
        }
        self.variables.push(RuntimeVariable {
            path: step.path.clone(),
            storage: step.storage,
            value,
            ordinal: step.ordinal,
        });
        self.runtime_variable_indices
            .insert(step.path.clone(), self.variables.len() - 1);
        if step.storage != StorageClass::Instance {
            let field = if step.storage == StorageClass::Global && entry.owner.is_none() {
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
        &mut self,
        entry: &VariableEntry,
        step: &InitializationStep,
        state: &mut ExecutionState,
    ) -> Result<(), RuntimeImageError> {
        if step.storage == StorageClass::Instance {
            return Ok(());
        }
        let field = if step.storage == StorageClass::Global && entry.owner.is_none() {
            variable_field(&step.path)?
        } else {
            FieldName::static_storage(&step.path)
        };
        let index = self
            .runtime_variable_indices
            .get(&step.path)
            .copied()
            .ok_or_else(|| RuntimeImageError::MissingInitializer(step.path.clone()))?;
        let value = self.variables[index].value.clone();
        self.stats.indexed_runtime_variable_reads += 1;
        state.set_global(field, value);
        Ok(())
    }

    fn execute_dynamic_initializer(
        &mut self,
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
        &mut self,
        entry: &VariableEntry,
    ) -> Result<BTreeMap<String, InitializerBinding>, DynamicInitializerFailure> {
        let initializer = entry
            .initializer
            .as_ref()
            .ok_or_else(|| DynamicInitializerFailure {
                phase: InitializerFailurePhase::Lowering,
                message: format!("initializer is absent for {:?}", entry.path),
                expanded_span: entry.span,
            })?;
        let references = initializer_binding_references(&initializer.tokens);
        let mut bindings = BTreeMap::new();
        let mut lookups = 0usize;

        for name in &references.bare {
            lookups += 1;
            if let Some(field) = self.binding_index.globals.get(name) {
                bindings.insert(name.clone(), InitializerBinding::Global(field.clone()));
            }
        }
        for (receiver, name) in &references.qualified {
            lookups += 1;
            let storage = self
                .global_types
                .get(receiver)
                .and_then(|type_path| self.shared_fields.get(type_path))
                .and_then(|fields| {
                    FieldName::parse(name)
                        .ok()
                        .and_then(|name| fields.get(&name))
                });
            if let Some(storage) = storage {
                bindings.insert(
                    format!("{receiver}.{name}"),
                    InitializerBinding::Global(storage.clone()),
                );
            }
        }
        for builtin in ["type", "parent_type"] {
            if references.bare.contains(builtin) {
                let field = FieldName::parse(builtin).expect("built-in datum field is valid");
                bindings.insert(builtin.to_owned(), InitializerBinding::SrcField(field));
            }
        }

        if let Some(owner) = &entry.owner
            && let Ok(owner) = TypePath::parse(&owner.path)
            && let Some(fields) = self.shared_fields.get(&owner)
        {
            for name in &references.bare {
                lookups += 1;
                if let Ok(field) = FieldName::parse(name)
                    && let Some(storage) = fields.get(&field)
                {
                    bindings.insert(name.clone(), InitializerBinding::Global(storage.clone()));
                }
            }
        }
        if entry.storage != StorageClass::Instance {
            self.stats.initializer_binding_index_lookups += lookups;
            self.stats.initializer_bindings_emitted += bindings.len();
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
                for name in &references.bare {
                    lookups += 1;
                    if let Some(field) = fields.get(name) {
                        bindings.insert(name.clone(), InitializerBinding::SrcField(field.clone()));
                    }
                }
            }
        }
        self.stats.initializer_binding_index_lookups += lookups;
        self.stats.initializer_bindings_emitted += bindings.len();
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

fn normalize_engine_instance_value(owner: &TypePath, field: &FieldName, value: Value) -> Value {
    if owner.as_str() != "/world" || field.as_str() != "view" {
        return value;
    }
    let Value::Text(text) = &value else {
        return value;
    };
    let Some((width, height)) = text.split_once('x').or_else(|| text.split_once('X')) else {
        return value;
    };
    let (Ok(width), Ok(height)) = (width.trim().parse::<u32>(), height.trim().parse::<u32>())
    else {
        return value;
    };
    if width == height && width > 0 && width % 2 == 1 {
        return Value::number((width.saturating_sub(1) / 2) as f32);
    }
    value
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
    let datum = TypePath::parse("/datum").expect("built-in datum path is valid");
    if types.contains_key(&datum) {
        for runtime_type in types.values_mut() {
            let path = runtime_type.path.as_str();
            if runtime_type.parent.is_none() && path != "/datum" && path != "/world" {
                runtime_type.parent = Some(datum.clone());
            }
        }
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
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::CompilerDatabase;
    use dm_globals::{StorageClass, UnsupportedCategory, VariableRegistry};
    use dm_lexer::{TokenKind, lex};
    use dm_semantics::standard_instance_field_names;
    use dm_value::{FieldName, TypePath, Value};
    use dm_vm::{compile_initializer, compile_module, execute_module_in_state};

    use super::{
        ConstantFieldApplication, InitializerFailurePhase, InitializerProcedureFrontier,
        RuntimeImage, RuntimeImageConstructionPhase, RuntimeImageError, builtin_client_defaults,
        builtin_mob_defaults, world_clock_values,
    };

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn byond_world_clocks_share_a_deterministic_instant_and_local_offset() {
        const BYOND_EPOCH_UNIX_MILLIS: i64 = 946_684_800_000;
        let clock = world_clock_values(
            BYOND_EPOCH_UNIX_MILLIS + 123_456_700,
            23,
            59,
            59,
            900,
            -19_800,
        );
        assert_eq!(clock.realtime_deciseconds, 1_234_567.0);
        assert_eq!(clock.timeofday_deciseconds, 863_999.0);
        assert_eq!(clock.timezone_hours, -5.5);
    }

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
    fn procedure_static_locals_survive_runtime_image_phase_transfer() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "var/global/seed = 1\n");
        let mut image = fixture.image();
        let syntax = dm_syntax::parse(
            "/proc/get_protected(list/list_ref)\n\tvar/static/list/protected_lists\n\tif(list_ref)\n\t\tprotected_lists = list_ref\n\treturn protected_lists\n/proc/seed_storage()\n\tget_protected(list())\n/proc/update_storage()\n\tvar/list/protected = get_protected()\n\tprotected[\"ADMIN\"] = list(1, 2)\n\treturn protected[\"ADMIN\"]\n",
        )
        .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();

        let mut first_phase = image.take_execution_state();
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/seed_storage").unwrap(),
            &[],
            &mut first_phase,
        )
        .unwrap();
        image.restore_execution_state(first_phase);

        let mut second_phase = image.take_execution_state();
        let result = execute_module_in_state(
            &module,
            module.procedure_id("/proc/update_storage").unwrap(),
            &[],
            &mut second_phase,
        )
        .expect("procedure-static list should survive the phase boundary");
        assert!(matches!(result, Value::List(_)));
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
    fn typed_null_global_receiver_reads_owner_shared_slots_in_static_initializer() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "var/global/datum/controller/global_vars/GLOB\n/datum/controller/global_vars\n\tvar/global/list/common = list(1)\n\tvar/global/list/rare = list(2)\n/datum/controller/royale\n\tvar/static/list/loot = list(\"common\" = GLOB.common, \"rare\" = GLOB.rare)\n",
        );
        let image = fixture.image();
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
        let loot = image
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with("/loot"))
            .expect("loot static");
        let Value::List(loot) = loot.value else {
            panic!("loot must be list");
        };
        let values = image.heap().list(loot).unwrap();
        for name in ["common", "rare"] {
            let expected = &image
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with(&format!("/{name}")))
                .unwrap()
                .value;
            assert!(
                values
                    .get_key(&Value::text(name))
                    .is_ok_and(|value| value.semantic_eq(expected))
            );
        }
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
        let Value::Datum(first_datum) = &first.value else {
            unreachable!("checked above")
        };
        assert_eq!(
            image
                .heap()
                .datum(*first_datum)
                .and_then(|datum| datum.field(&field("value"))),
            Ok(&Value::number(8.0)),
            "global new must see the completed inherited-default snapshot"
        );
        assert_eq!(
            image.stats().execution_metadata_builds,
            1,
            "the whole type tree should be snapshotted exactly once"
        );
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
        assert_eq!(image.stats().initializer_module_deferred_procedures, 1);
        assert_eq!(
            image.stats().initializer_module_materialized_procedures,
            1,
            "the initializer's reachable callee should lower exactly once"
        );
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
    }

    #[test]
    fn global_redeclaration_updates_and_syncs_through_exact_path_index() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/var/global/value\n/var/value = 7\n/var/global/copy = value\n",
        );
        let image = fixture.image();
        assert_eq!(
            image
                .variable("/var/value")
                .and_then(|variable| variable.value.as_number()),
            Some(7.0)
        );
        assert_eq!(
            image
                .variable("/var/copy")
                .and_then(|variable| variable.value.as_number()),
            Some(7.0),
            "the indexed update must be visible in the shared initializer state"
        );
        assert_eq!(image.stats().indexed_runtime_variable_updates, 1);
        assert_eq!(image.stats().indexed_runtime_variable_reads, 2);
    }

    #[test]
    fn indirect_call_initializer_falls_back_to_complete_symbol_inventory() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/proc/reached()\n\treturn 7\n/proc/unrelated()\n\treturn 99\n/var/global/target = /proc/reached\n/var/global/result = call(target)()\n",
        );
        let image = fixture.image();
        assert_eq!(
            image
                .variable("/var/result")
                .and_then(|variable| variable.value.as_number()),
            Some(7.0)
        );
        assert_eq!(image.stats().initializer_complete_symbol_inventory, 1);
        assert_eq!(image.stats().initializer_module_deferred_procedures, 2);
        assert_eq!(
            image.stats().initializer_module_materialized_procedures,
            1,
            "the complete inventory remains body-lazy"
        );
    }

    #[test]
    fn typed_initializer_construction_narrows_new_frontier_and_calls_inherited_override() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/marker\n\tNew(value)\n\t\tmarker = value\n/datum/base/child\n\tNew(value)\n\t\t..()\n\t\tmarker += 1\n/datum/base/child/grandchild\n/datum/unrelated\n\tNew()\n\t\treturn 99\n/var/global/datum/base/child/explicit = new /datum/base/child(4)\n/var/global/datum/base/child/grandchild/inferred = new(5)\n",
        );
        let image = fixture.image();
        let marker = |path: &str| {
            let Value::Datum(datum) = &image.variable(path).expect("constructed global").value
            else {
                panic!("{path} should contain a datum");
            };
            image
                .heap()
                .datum_field(*datum, &field("marker"))
                .expect("marker")
                .as_number()
        };
        assert_eq!(marker("/var/explicit"), Some(5.0));
        assert_eq!(marker("/var/inferred"), Some(6.0));
        assert_eq!(image.stats().initializer_typed_constructor_targets, 2);
        assert_eq!(image.stats().initializer_dynamic_constructor_frontier, 0);
        assert_eq!(
            image.stats().initializer_module_deferred_procedures,
            2,
            "only child New and its exact parent-call target should be retained"
        );
        assert_eq!(image.stats().initializer_module_materialized_procedures, 2);
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
    fn square_text_world_view_materializes_as_byond_numeric_radius() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "/world\n\tview = \"15x15\"\n");
        let mut image = fixture.image();
        let world = image
            .allocate_datum(&type_path("/world"))
            .expect("world should allocate");
        assert_eq!(
            image.heap().datum_field(world, &field("view")),
            Ok(&Value::number(7.0)),
            "BYOND 516 exposes a 15x15 square view as radius 7"
        );
        let state = image.take_execution_state();
        assert_eq!(
            state.initial_value(&type_path("/world"), &field("view")),
            Some(&Value::number(7.0)),
            "initial(world.view) and the live engine field must agree"
        );
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
    fn dynamic_initializer_construction_retains_all_new_candidates() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/one\n\tvar/marker\n\tNew()\n\t\tmarker = 1\n/datum/two\n\tNew()\n\t\treturn 2\n/var/global/type_to_make = /datum/one\n/var/global/result = new type_to_make()\n",
        );
        let image = fixture.image();
        let Value::Datum(result) = &image.variable("/var/result").expect("result").value else {
            panic!("dynamic new should construct a datum");
        };
        assert_eq!(
            image
                .heap()
                .datum_field(*result, &field("marker"))
                .expect("marker"),
            &Value::number(1.0)
        );
        assert_eq!(image.stats().initializer_dynamic_constructor_frontier, 1);
        assert_eq!(image.stats().initializer_module_deferred_procedures, 2);
        assert_eq!(image.stats().initializer_module_materialized_procedures, 1);
    }

    #[test]
    fn inherited_runtime_default_yields_to_descendant_null_before_new() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/map_template/shuttle\n\tvar/list/who_can_purchase = list(\"Captain\")\n\tvar/occupancy_limit\n/datum/map_template/shuttle/emergency\n\tNew()\n\t\tif(!occupancy_limit && who_can_purchase)\n\t\t\tCRASH(\"purchasable shuttle needs an occupancy limit\")\n/datum/map_template/shuttle/emergency/backup\n\twho_can_purchase = null\n/datum/map_template/shuttle/emergency/unlisted\n\tvar/who_can_purchase\n/datum/map_template/shuttle/emergency/for_sale\n\twho_can_purchase = list(\"Chief Engineer\")\n\toccupancy_limit = \"50\"\n/datum/interleave\n\tvar/list/value = list(\"first\")\n/datum/interleave\n\tvar/value\n/datum/interleave\n\tvalue = list(\"last\")\n/datum/side_effect\n\tvar/a = (b = 99)\n\tvar/b = 7\n/var/global/datum/map_template/shuttle/emergency/backup/vm_backup = new /datum/map_template/shuttle/emergency/backup\n/var/global/datum/map_template/shuttle/emergency/unlisted/vm_unlisted = new /datum/map_template/shuttle/emergency/unlisted\n/var/global/datum/map_template/shuttle/emergency/for_sale/vm_for_sale = new /datum/map_template/shuttle/emergency/for_sale\n/var/global/datum/interleave/vm_interleave = new /datum/interleave\n/var/global/datum/side_effect/vm_side_effect = new /datum/side_effect\n",
        );

        let mut image = fixture.image();
        let read_global_field = |image: &RuntimeImage, suffix: &str, name: &str| {
            let Value::Datum(datum) = image
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with(suffix))
                .unwrap_or_else(|| panic!("missing global {suffix}"))
                .value
            else {
                panic!("{suffix} should hold a datum")
            };
            image
                .heap()
                .datum_field(datum, &field(name))
                .unwrap_or_else(|error| panic!("missing {name} on {suffix}: {error}"))
                .clone()
        };

        assert_eq!(
            read_global_field(&image, "/vm_backup", "who_can_purchase"),
            Value::Null,
            "VM dynamic new must apply the backup subtype's explicit null after the inherited list"
        );
        assert_eq!(
            read_global_field(&image, "/vm_unlisted", "who_can_purchase"),
            Value::Null,
            "an explicit uninitialized descendant declaration is also a null override"
        );
        assert!(matches!(
            read_global_field(&image, "/vm_for_sale", "who_can_purchase"),
            Value::List(_)
        ));
        let Value::List(interleaved) = read_global_field(&image, "/vm_interleave", "value") else {
            panic!("the final same-owner runtime initializer must follow the null declaration")
        };
        assert_eq!(
            image.heap().list(interleaved).unwrap().get(1),
            Ok(&Value::text("last")),
            "VM catalog actions must retain same-owner source order"
        );
        assert_eq!(
            read_global_field(&image, "/vm_side_effect", "b"),
            Value::number(7.0),
            "a later constant must follow an earlier initializer's cross-field side effect"
        );

        for path in [
            "/datum/map_template/shuttle/emergency/backup",
            "/datum/map_template/shuttle/emergency/unlisted",
        ] {
            let datum = image
                .allocate_datum(&type_path(path))
                .expect("host allocation should preserve descendant null ordering");
            assert_eq!(
                image.heap().datum_field(datum, &field("who_can_purchase")),
                Ok(&Value::Null),
                "host allocation must apply {path}'s null after the inherited list"
            );
        }
        let side_effect = image
            .allocate_datum(&type_path("/datum/side_effect"))
            .expect("host allocation should preserve cross-field source order");
        assert_eq!(
            image.heap().datum_field(side_effect, &field("b")),
            Ok(&Value::number(7.0))
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
            "/datum/child\n\tvar/list/nested_items = list(9)\n/datum/base\n\tvar/list/items = list(1)\n\tvar/datum/child/child = new /datum/child\n/datum/base/sub\n\titems = list(2)\n\tchild = new /datum/child\n",
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
            "/datum/child"
        );
        let first_nested = image
            .heap()
            .datum_field(first_child, &field("nested_items"))
            .expect("nested new must execute its own instance defaults");
        let second_nested = image
            .heap()
            .datum_field(second_child, &field("nested_items"))
            .expect("second nested new must execute its own instance defaults");
        assert!(matches!(first_nested, Value::List(_)));
        assert!(matches!(second_nested, Value::List(_)));
        assert_ne!(first_nested, second_nested, "nested lists must not alias");
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
    fn released_preflight_caches_rebuild_lazily_without_changing_defaults() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/list/items = list(7)\n/datum/base/sub\n",
        );
        let mut image = fixture.image();
        let subtype = type_path("/datum/base/sub");
        image
            .preflight_instance_initializers([subtype.clone()])
            .expect("valid plan should preflight");
        let released = image.release_allocation_caches();
        assert_eq!(released.initializer_plans, 1);
        assert_eq!(released.initializer_programs, 1);
        assert_eq!(released.allocation_plans, 1);

        let datum = image
            .allocate_datum(&subtype)
            .expect("allocation should lazily rebuild released plans");
        let Value::List(items) = image
            .heap()
            .datum_field(datum, &field("items"))
            .expect("dynamic inherited default should still materialize")
        else {
            panic!("items should remain a fresh list")
        };
        assert_eq!(
            image.heap().list(*items).unwrap().get(1),
            Ok(&Value::number(7.0))
        );
    }

    #[test]
    fn sibling_preflight_shares_compiled_ancestor_initializer_program() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/list/items = list(1)\n/datum/base/alpha\n/datum/base/beta\n",
        );
        let mut image = fixture.image();
        let stats = image
            .preflight_instance_initializers([
                type_path("/datum/base/beta"),
                type_path("/datum/base/alpha"),
            ])
            .expect("sibling plans should preflight");
        assert_eq!(stats.types, 2);
        assert_eq!(stats.plans_compiled, 2);
        assert_eq!(image.stats().instance_initializer_candidates_indexed, 1);
        assert_eq!(
            image.stats().instance_initializer_unique_programs_compiled,
            1,
            "the inherited source initializer must compile only once"
        );
        assert_eq!(image.stats().instance_initializer_plan_references, 2);

        let alpha = image
            .allocate_datum(&type_path("/datum/base/alpha"))
            .expect("alpha should allocate");
        let beta = image
            .allocate_datum(&type_path("/datum/base/beta"))
            .expect("beta should allocate");
        let alpha_items = image
            .heap()
            .datum_field(alpha, &field("items"))
            .expect("alpha items");
        let beta_items = image
            .heap()
            .datum_field(beta, &field("items"))
            .expect("beta items");
        assert!(matches!(alpha_items, Value::List(_)));
        assert!(matches!(beta_items, Value::List(_)));
        assert_ne!(
            alpha_items, beta_items,
            "sharing compiled code must not share per-instance list values"
        );
        assert_eq!(
            image.stats().instance_initializer_unique_programs_compiled,
            1,
            "allocation must reuse the preflighted shared program"
        );
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
        let lists_before = image.heap().live_list_count();
        let datum_id = image
            .allocate_datum(&type_path("/obj/example"))
            .expect("atom subtype should allocate");
        assert_eq!(
            image.heap().live_list_count(),
            lists_before + 1,
            "an untouched atom must allocate only its spatial contents list"
        );
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
        for name in [
            "pixel_x",
            "pixel_y",
            "pixel_w",
            "pixel_z",
            "maptext_x",
            "maptext_y",
        ] {
            assert_eq!(
                datum.field(&field(name)).unwrap().as_number(),
                Some(0.0),
                "{name} should have its engine-owned atom default",
            );
        }
        assert_eq!(datum.field(&field("transform")), Ok(&Value::Null));
        assert!(matches!(
            datum.field(&field("contents")),
            Ok(Value::List(_))
        ));
        for name in [
            "filters",
            "overlays",
            "underlays",
            "verbs",
            "vis_contents",
            "vis_locs",
        ] {
            assert!(
                !matches!(datum.field(&field(name)), Ok(Value::List(_))),
                "{name} must not allocate a list before the VM observes it",
            );
        }

        let second = image
            .allocate_datum(&type_path("/obj/example"))
            .expect("second atom subtype should allocate");
        assert_eq!(
            image.heap().live_list_count(),
            lists_before + 2,
            "native appearance and visibility lists must not scale with untouched atoms"
        );
        assert_ne!(
            image.heap().datum_field(datum_id, &field("contents")),
            image.heap().datum_field(second, &field("contents")),
            "eager spatial contents lists must not alias between instances",
        );

        let mob = image.allocate_datum(&type_path("/mob")).unwrap();
        assert_eq!(
            image.heap().datum_field(mob, &field("see_in_dark")),
            Ok(&Value::number(2.0))
        );
        let client = image.allocate_datum(&type_path("/client")).unwrap();
        for name in ["images", "screen", "verbs"] {
            assert!(matches!(
                image.heap().datum_field(client, &field(name)),
                Ok(Value::List(_))
            ));
        }
    }

    #[test]
    fn dynamic_dummy_mob_exposes_null_client_during_initialize_add_verb() {
        const PROCEDURES: &str = "/proc/add_verb(mob/target)\n\tif(!target.client)\n\t\treturn isnull(target.client)\n\treturn 0\n/mob/living/carbon/human/dummy/proc/Initialize()\n\treturn add_verb(src)\n/mob/living/carbon/human/dummy/New()\n\tglobal.add_verb_result = Initialize()\n/proc/run()\n\tnew /mob/living/carbon/human/dummy\n\treturn global.add_verb_result\n";
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            &format!("var/global/add_verb_result = 0\n{PROCEDURES}"),
        );
        let mut image = fixture.image();
        let syntax = dm_syntax::parse(PROCEDURES).expect("dummy mob fixture should parse");
        let module = compile_module(&syntax.definitions).expect("dummy mob fixture should lower");
        let mob = image
            .allocate_datum(&type_path("/mob"))
            .expect("mob should allocate");
        let client = image
            .allocate_datum(&type_path("/client"))
            .expect("client should allocate");
        let mut state = image.take_execution_state();
        let dummy = type_path("/mob/living/carbon/human/dummy");

        for (path, datum, defaults) in [
            (
                type_path("/mob"),
                mob,
                builtin_mob_defaults().into_iter().collect::<Vec<_>>(),
            ),
            (
                type_path("/client"),
                client,
                builtin_client_defaults().into_iter().collect::<Vec<_>>(),
            ),
        ] {
            let semantic_fields = standard_instance_field_names(path.as_str());
            for (name, expected) in defaults {
                assert!(
                    semantic_fields.contains(&name),
                    "materialized engine field {path}.{name} must be in the semantic catalog",
                );
                assert_eq!(
                    state.heap().datum_field(datum, &field(name)),
                    Ok(&expected),
                    "unexpected materialized engine default for {path}.{name}",
                );
                assert_eq!(
                    state.initial_value(&path, &field(name)),
                    Some(&expected),
                    "transferred initial metadata must retain {path}.{name}",
                );
            }
        }
        assert_eq!(
            state.initial_value(&dummy, &field("client")),
            Some(&Value::Null)
        );
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/run").expect("run procedure"),
                &[],
                &mut state,
            )
            .expect("headless dummy Initialize should read mob.client"),
            Value::number(1.0),
        );
    }

    #[test]
    fn semantic_atom_and_movable_engine_field_catalogs_are_materialized() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/obj/example\n/particles/example\n/mutable_appearance/example\n",
        );
        let mut image = fixture.image();
        let object = image
            .allocate_datum(&type_path("/obj/example"))
            .expect("object should allocate");

        for type_path in ["/atom", "/atom/movable"] {
            for name in standard_instance_field_names(type_path) {
                if [
                    "filters",
                    "overlays",
                    "underlays",
                    "verbs",
                    "vis_contents",
                    "vis_locs",
                ]
                .contains(&name)
                {
                    assert!(
                        !matches!(
                            image.heap().datum_field(object, &field(name)),
                            Ok(Value::List(_))
                        ),
                        "semantic engine field {type_path}.{name} must remain lazy",
                    );
                    continue;
                }
                assert!(
                    image.heap().datum_field(object, &field(name)).is_ok(),
                    "semantic engine field {type_path}.{name} must have runtime storage",
                );
            }
        }
        let particles = image
            .allocate_datum(&type_path("/particles/example"))
            .expect("particles should allocate");
        for name in standard_instance_field_names("/particles") {
            assert!(
                image.heap().datum_field(particles, &field(name)).is_ok(),
                "semantic particle field {name} must have runtime storage",
            );
        }
        let appearance = image
            .allocate_datum(&type_path("/mutable_appearance/example"))
            .expect("mutable appearance should allocate");
        for name in standard_instance_field_names("/image") {
            assert!(
                image.heap().datum_field(appearance, &field(name)).is_ok(),
                "semantic image field {name} must have runtime storage",
            );
        }
        for (name, expected) in [
            ("appearance", Value::Null),
            ("desc", Value::Null),
            ("gender", Value::text("neuter")),
            ("luminosity", Value::number(0.0)),
            ("particles", Value::Null),
            ("suffix", Value::Null),
            ("step_x", Value::number(0.0)),
            ("step_y", Value::number(0.0)),
        ] {
            assert_eq!(
                image.heap().datum_field(object, &field(name)),
                Ok(&expected),
                "unexpected BYOND default for {name}",
            );
        }
    }

    #[test]
    fn parentless_engine_types_inherit_base_datum_storage() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum\n\tvar/datum_flags = 0\n/regex/example\n/icon/example\n",
        );
        let mut image = fixture.image();
        for path in ["/regex/example", "/icon/example"] {
            let datum = image.allocate_datum(&type_path(path)).unwrap();
            assert_eq!(
                image.heap().datum_field(datum, &field("datum_flags")),
                Ok(&Value::number(0.0)),
                "{path} must inherit /datum fields",
            );
        }
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
    fn linked_datum_expression_appends_without_renumbering_existing_entries() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "var/global/offset = 3\n/datum/base\n\tvar/base = 4\n",
        );
        let mut image = fixture.image();
        let datum = image
            .allocate_datum(&type_path("/datum/base"))
            .expect("datum should allocate");
        let mut state = image.take_execution_state();

        let seed_tokens = lex("1")
            .unwrap()
            .into_iter()
            .filter(|token| {
                !matches!(
                    token.kind,
                    TokenKind::LineStart { .. } | TokenKind::Newline | TokenKind::LineContinuation
                )
            })
            .collect::<Vec<_>>();
        let seed = compile_initializer(&seed_tokens, &BTreeMap::new(), None).unwrap();
        let mut module = seed.module().clone();
        let original = module.procedure_id_at(0).expect("seed entry");

        assert_eq!(
            image
                .evaluate_datum_expression_linked(datum, "base + offset", &mut state, &mut module,)
                .expect("linked expression should execute"),
            Value::number(7.0)
        );
        assert_eq!(module.procedure_id_at(0), Some(original));
        let appended = module.procedure_id_at(1).expect("appended entry");
        assert_eq!(module.procedure_path(appended), Some("<initializer>"));
        assert!(module.procedure(appended).is_some_and(|program| {
            program
                .source_spans
                .iter()
                .all(|span| span.end > span.start)
        }));
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
    fn subsystem_typepath_initial_order_inherits_and_overrides_for_sorting() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/controller/subsystem\n\tvar/init_order = 40\n/datum/controller/subsystem/inherited\n/datum/controller/subsystem/overridden\n\tinit_order = 90\n",
        );
        let mut image = fixture.image();
        let state = image.take_execution_state();
        let base = type_path("/datum/controller/subsystem");
        let inherited = type_path("/datum/controller/subsystem/inherited");
        let overridden = type_path("/datum/controller/subsystem/overridden");
        let init_order = field("init_order");

        assert_eq!(
            state.initial_value(&base, &init_order),
            Some(&Value::number(40.0))
        );
        assert_eq!(
            state.initial_value(&inherited, &init_order),
            Some(&Value::number(40.0))
        );
        assert_eq!(
            state.initial_value(&overridden, &init_order),
            Some(&Value::number(90.0))
        );

        // cmp_subsystem_init(a, b) is `initial(b.init_order) -
        // initial(a.init_order)`, so the override sorts before the inherited
        // parent value while an unmodified subtype compares equal to it.
        let compare = |a: &TypePath, b: &TypePath| {
            state
                .initial_value(b, &init_order)
                .and_then(Value::as_number)
                .unwrap_or_default()
                - state
                    .initial_value(a, &init_order)
                    .and_then(Value::as_number)
                    .unwrap_or_default()
        };
        assert_eq!(compare(&base, &inherited), 0.0);
        assert_eq!(compare(&inherited, &overridden), 50.0);
        assert_eq!(compare(&overridden, &inherited), -50.0);
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

    #[test]
    fn construction_observer_emits_deterministic_phase_boundaries() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/proc/value()\n\treturn 7\n/datum/base\n\tvar/list/items = list(1)\n/var/global/result = value()\n",
        );
        let compilation = CompilerDatabase::new()
            .compile(fixture.0.join("world.dme"))
            .expect("fixture should compile");
        let mut events = Vec::new();
        RuntimeImage::from_compilation_with_observer(&compilation, |event| {
            events.push((event.phase, event.completed));
        })
        .expect("image should build");
        let phases = [
            RuntimeImageConstructionPhase::VariableRegistry,
            RuntimeImageConstructionPhase::TypeInventory,
            RuntimeImageConstructionPhase::InstanceConstants,
            RuntimeImageConstructionPhase::ExecutionMetadata,
            RuntimeImageConstructionPhase::ProcedureRegistry,
            RuntimeImageConstructionPhase::InitializerModuleLink,
            RuntimeImageConstructionPhase::InstanceInitializerCompilation,
            RuntimeImageConstructionPhase::GlobalInitializerExecution,
            RuntimeImageConstructionPhase::Finalization,
        ];
        assert_eq!(
            events,
            phases
                .into_iter()
                .flat_map(|phase| [(phase, false), (phase, true)])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn initializer_binding_work_depends_on_references_not_global_inventory() {
        let build = |unused_globals: usize| {
            let fixture = Fixture::new();
            fixture.write("world.dme", "#include \"types.dm\"\n");
            let mut source =
                "var/global/used = 7\n/datum/base\n\tvar/list/items = list(used)\n".to_owned();
            for index in 0..unused_globals {
                source.push_str(&format!("var/global/unused_{index} = {index}\n"));
            }
            fixture.write("types.dm", &source);
            let image = fixture.image();
            (
                image.stats().initializer_binding_index_lookups,
                image.stats().initializer_bindings_emitted,
            )
        };
        assert_eq!(
            build(2),
            build(200),
            "unreferenced globals must not increase initializer binding work"
        );
    }

    #[test]
    fn initializer_frontier_uses_call_syntax_not_unrelated_identifiers() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/proc/real_call()\n/proc/ref_target()\n/proc/text_target()\n/datum/holder/proc/member_call()\n/var/global/unrelated_identifier = 1\n/var/global/datum/holder/holder\n/var/global/list/test = list(unrelated_identifier, /datum/noise, \"ordinary_string\", real_call(), holder.member_call(), /proc/ref_target, text2path(\"/proc/text_target\"))\n",
        );
        let compilation = CompilerDatabase::new()
            .compile(fixture.0.join("world.dme"))
            .expect("fixture should compile");
        let registry = VariableRegistry::build(&compilation);
        let entry = registry
            .entries()
            .iter()
            .find(|entry| entry.path == "/var/test")
            .expect("test initializer");
        let mut frontier = InitializerProcedureFrontier::default();
        frontier.include(&compilation, entry);

        for selector in [
            "list",
            "real_call",
            "member_call",
            "ref_target",
            "text2path",
            "text_target",
        ] {
            assert!(frontier.selectors.contains(selector), "missing {selector}");
        }
        for false_selector in [
            "unrelated_identifier",
            "datum",
            "noise",
            "holder",
            "ordinary_string",
        ] {
            assert!(
                !frontier.selectors.contains(false_selector),
                "non-call identifier {false_selector} must not become a procedure root"
            );
        }
        assert!(!frontier.requires_complete_inventory);
    }

    #[test]
    fn initializer_frontier_preserves_indirect_call_fallback() {
        let fixture = Fixture::new();
        fixture.write(
            "world.dme",
            "/var/global/target\n/var/global/result = call(target)()\n",
        );
        let compilation = CompilerDatabase::new()
            .compile(fixture.0.join("world.dme"))
            .expect("fixture should compile");
        let registry = VariableRegistry::build(&compilation);
        let entry = registry
            .entries()
            .iter()
            .find(|entry| entry.path == "/var/result")
            .expect("result initializer");
        let mut frontier = InitializerProcedureFrontier::default();
        frontier.include(&compilation, entry);
        assert!(frontier.requires_complete_inventory);
    }
}
