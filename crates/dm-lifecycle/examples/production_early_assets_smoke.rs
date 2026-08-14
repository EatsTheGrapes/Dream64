//! Bounded production gate for Monkestation's real Early Assets subsystem.
//!
//! This runs the real global/Genesis bootstrap and then invokes only
//! `SSearly_assets.Initialize()`. It deliberately skips map allocation and
//! Mapping so antagonist/species preview failures can be iterated quickly.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::CompilerDatabase;
use dm_runtime::RuntimeImage;
use dm_semantics::{ExecutableProcedures, ProcedureImplementationId, ProcedureRegistry};
use dm_value::{DatumId, FieldName, Value};
use dm_vm::{
    ExecutionContext, ExecutionLimits, ExecutionState, advance_scheduler,
    execute_module_with_limits_in_context,
};

const GENESIS: &str = "/world/proc/Genesis";
const CONFIG_LOAD: &str = "/datum/controller/configuration/proc/Load";
const ID_ACCESS: &str = "/datum/controller/subsystem/id_access/proc/Initialize";
const REAGENTS: &str = "/datum/controller/subsystem/processing/reagents/proc/Initialize";
const RESEARCH: &str = "/datum/controller/subsystem/research/proc/Initialize";
const GREYSCALE: &str = "/datum/controller/subsystem/processing/greyscale/proc/Initialize";
const EARLY_ASSETS: &str = "/datum/controller/subsystem/early_assets/proc/Initialize";
const FLUIDS: &str = "/datum/controller/subsystem/fluids/proc/Initialize";
const GENERATE_HOLOMAP: &str = "/datum/controller/subsystem/holomaps/proc/generate_holomap";
const ATOMS: &str = "/datum/controller/subsystem/atoms/proc/Initialize";
const KIRBY_INITIALIZE: &str = "/obj/item/kirbyplants/random/proc/Initialize";
const JUKEBOX_INITIALIZE: &str = "/obj/machinery/jukebox/proc/Initialize";
const RUNE_SPAWN_INITIALIZE: &str = "/obj/effect/temp_visual/cult/rune_spawn/proc/Initialize";
const BLOOD_PACK_INITIALIZE: &str = "/obj/item/reagent_containers/blood/proc/Initialize";
const MACHINES: &str = "/datum/controller/subsystem/machines/proc/Initialize";
const AIR: &str = "/datum/controller/subsystem/air/proc/Initialize";
const PERSISTENCE: &str = "/datum/controller/subsystem/persistence/proc/Initialize";
const PERSISTENT_PAINTINGS: &str =
    "/datum/controller/subsystem/persistent_paintings/proc/Initialize";
const ASSETS: &str = "/datum/controller/subsystem/assets/proc/Initialize";
const STATION_COLORING: &str = "/datum/controller/subsystem/station_coloring/proc/Initialize";
const ICON_SMOOTH: &str = "/datum/controller/subsystem/icon_smooth/proc/Initialize";
const LIGHTING: &str = "/datum/controller/subsystem/lighting/proc/Initialize";
const SHUTTLE: &str = "/datum/controller/subsystem/shuttle/proc/Initialize";
const CREDITS: &str = "/datum/controller/subsystem/credits/proc/Initialize";
const INIT_PROFILER: &str = "/datum/controller/subsystem/init_profiler/proc/Initialize";
const BECOME_AREA_SENSITIVE: &str = "/atom/movable/proc/become_area_sensitive";
const DCS_GET_ID: &str = "/datum/controller/subsystem/processing/dcs/proc/GetIdFromArguments";
const DCS_GET_ELEMENT: &str = "/datum/controller/subsystem/processing/dcs/proc/GetElement";
const ADD_ELEMENT: &str = "/datum/proc/_AddElement";
const DECAL_SMOOTH_REACT: &str = "/datum/element/decal/proc/smooth_react";

fn main() -> ExitCode {
    std::thread::Builder::new()
        .name("production-early-assets-smoke".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .expect("failed to create production early-assets smoke thread")
        .join()
        .unwrap_or_else(|_| Err("production early-assets smoke thread panicked".to_owned()))
        .map_or_else(
            |error| {
                eprintln!("early-assets-smoke: FAIL: {error}");
                ExitCode::FAILURE
            },
            |()| ExitCode::SUCCESS,
        )
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    let environment = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: production_early_assets_smoke <world.dme>")?,
    );
    if arguments.next().is_some() {
        return Err("usage: production_early_assets_smoke <world.dme>".to_owned());
    }
    let project_root = environment
        .parent()
        .ok_or_else(|| format!("project has no parent: {}", environment.display()))?
        .to_path_buf();
    let started = Instant::now();
    let cache = project_cache_file(&environment);
    let (compilation, cache_hit) = CompilerDatabase::new()
        .compile_cached(&environment, &cache)
        .map_err(|error| format!("project compilation failed: {error}"))?;
    eprintln!(
        "early-assets-smoke: project ready elapsed_ms={} preprocessing_cache={}",
        started.elapsed().as_millis(),
        if cache_hit { "hit" } else { "miss" },
    );

    let procedures = ProcedureRegistry::build(&compilation);
    let roots = [
        GENESIS,
        CONFIG_LOAD,
        ID_ACCESS,
        REAGENTS,
        RESEARCH,
        GREYSCALE,
        EARLY_ASSETS,
        FLUIDS,
        GENERATE_HOLOMAP,
        ATOMS,
        KIRBY_INITIALIZE,
        JUKEBOX_INITIALIZE,
        RUNE_SPAWN_INITIALIZE,
        BLOOD_PACK_INITIALIZE,
        MACHINES,
        AIR,
        PERSISTENCE,
        PERSISTENT_PAINTINGS,
        ASSETS,
        STATION_COLORING,
        ICON_SMOOTH,
        LIGHTING,
        SHUTTLE,
        CREDITS,
        INIT_PROFILER,
        BECOME_AREA_SENSITIVE,
        DCS_GET_ID,
        DCS_GET_ELEMENT,
        ADD_ELEMENT,
        DECAL_SMOOTH_REACT,
    ]
    .map(|path| effective_target(&procedures, path))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    let mut executable = procedures
        .compile_vm_implementations_symbolic_dynamic(&compilation, roots)
        .map_err(|error| format!("production lowering failed: {error}"))?;
    eprintln!(
        "early-assets-smoke: linked deferred={} elapsed_ms={}",
        executable.module().deferred_procedure_count(),
        started.elapsed().as_millis(),
    );
    if let Some(path) = env::var_os("DREAM64_DUMP_PROC") {
        let path = path.to_string_lossy();
        let target = effective_target(&procedures, &path)?;
        let entry = executable
            .implementation(target)
            .ok_or_else(|| format!("linked VM entry is missing: {path}"))?;
        let program = executable
            .module()
            .procedure(entry)
            .ok_or_else(|| format!("compiled VM body is missing: {path}"))?;
        eprintln!(
            "production-bytecode: parameters={:?}",
            program.parameter_names
        );
        for (index, instruction) in program.instructions.iter().enumerate() {
            eprintln!(
                "production-bytecode: {index:03} {:?} span={:?}",
                instruction,
                program.source_spans.get(index)
            );
        }
        return Ok(());
    }

    let mut runtime = RuntimeImage::from_compilation(&compilation)
        .map_err(|error| format!("runtime image construction failed: {error}"))?;
    let kirby_fixture = runtime
        .allocate_datum(
            &dm_value::TypePath::parse("/obj/item/kirbyplants/random/dead")
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("failed to allocate Kirby plant fixture: {error}"))?;
    let jukebox_fixture = runtime
        .allocate_datum(
            &dm_value::TypePath::parse("/obj/machinery/jukebox")
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("failed to allocate jukebox fixture: {error}"))?;
    let rune_spawn_fixture = runtime
        .allocate_datum(
            &dm_value::TypePath::parse("/obj/effect/temp_visual/cult/rune_spawn")
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("failed to allocate rune-spawn fixture: {error}"))?;
    let blood_pack_fixture = runtime
        .allocate_datum(
            &dm_value::TypePath::parse("/obj/item/reagent_containers/blood/oil")
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("failed to allocate blood-pack fixture: {error}"))?;
    let neon_carpet_fixture = runtime
        .allocate_datum(
            &dm_value::TypePath::parse("/turf/open/floor/carpet/neon/simple/blue/nodots")
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("failed to allocate neon-carpet fixture: {error}"))?;
    let world = runtime
        .canonical_world()
        .ok_or("canonical /world datum is missing")?;
    let mut state = runtime.take_execution_state();
    let processable = dm_value::TypePath::parse("/datum/element/processable")
        .map_err(|error| error.to_string())?;
    let hash_start = state
        .initial_value(&processable, &field("argument_hash_start_idx")?)
        .cloned();
    if hash_start != Some(Value::number(2.0)) {
        return Err(format!(
            "processable bespoke hash start should be 2, received {hash_start:?}"
        ));
    }
    let element_flags = state
        .initial_value(&processable, &field("element_flags")?)
        .cloned();
    if element_flags != Some(Value::number(2.0)) {
        return Err(format!(
            "processable element flags should be bespoke (2), received {element_flags:?}"
        ));
    }
    state.set_project_root(project_root);
    state.set_global(field("world")?, Value::Datum(world));
    // Map-generation and other host allocation paths can intentionally retain
    // only a datum's type/identity. Declared inherited fields must still read
    // through the production initial-value catalog before their first write.
    let sparse_ore_path =
        dm_value::TypePath::parse("/obj/item/stack/ore/gold").map_err(|error| error.to_string())?;
    if state.initial_value(&sparse_ore_path, &field("important_recursive_contents")?)
        != Some(&Value::Null)
    {
        return Err("gold ore is missing inherited important_recursive_contents metadata".into());
    }
    let sparse_goldgrub_path = dm_value::TypePath::parse("/mob/living/basic/mining/goldgrub")
        .map_err(|error| error.to_string())?;
    if state.initial_value(
        &sparse_goldgrub_path,
        &field("important_recursive_contents")?,
    ) != Some(&Value::Null)
    {
        return Err("goldgrub is missing inherited important_recursive_contents metadata".into());
    }
    // Model the map allocator for the containing turf. A bare heap datum is
    // intentionally only valid for the sparse nested fixtures below; turfs
    // require their inherited atom fields (including
    // `important_recursive_contents`) to be materialized before the real
    // area-sensitive procedure writes them.
    let containing_turf = state
        .allocate_compact_map_datum(
            dm_value::TypePath::parse("/turf/open/floor").map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("failed to allocate containing turf fixture: {error}"))?;
    for (name, value) in [
        ("x", Value::number(8.0)),
        ("y", Value::number(9.0)),
        ("z", Value::number(1.0)),
        ("loc", Value::Null),
        // `become_area_sensitive` appends to this inherited atom list. The
        // compact map datum retains sparse defaults, so make the catalog's
        // verified null default concrete for this direct heap-field path.
        ("important_recursive_contents", Value::Null),
    ] {
        state
            .heap_mut()
            .set_datum_field(containing_turf, field(name)?, value)
            .map_err(|error| error.to_string())?;
    }
    let sparse_goldgrub = state.heap_mut().allocate_datum(sparse_goldgrub_path);
    for (name, value) in [
        ("x", Value::number(0.0)),
        ("y", Value::number(0.0)),
        ("z", Value::number(0.0)),
        ("loc", Value::Datum(containing_turf)),
    ] {
        state
            .heap_mut()
            .set_datum_field(sparse_goldgrub, field(name)?, value)
            .map_err(|error| error.to_string())?;
    }
    let sparse_ore = state.heap_mut().allocate_datum(sparse_ore_path);
    for (name, value) in [
        ("x", Value::number(0.0)),
        ("y", Value::number(0.0)),
        ("z", Value::number(0.0)),
        ("loc", Value::Datum(sparse_goldgrub)),
    ] {
        state
            .heap_mut()
            .set_datum_field(sparse_ore, field(name)?, value)
            .map_err(|error| error.to_string())?;
    }
    invoke_with_args(
        &mut executable,
        &procedures,
        BECOME_AREA_SENSITIVE,
        sparse_ore,
        &[Value::text("production-smoke")],
        &mut state,
    )?;
    for datum in [sparse_ore, sparse_goldgrub, containing_turf] {
        state
            .heap_mut()
            .destroy_datum(datum)
            .map_err(|error| error.to_string())?;
    }
    // The full boot's map allocator normally materializes these engine-owned
    // world geometry fields before Genesis. This bounded gate has no map.
    for (name, value) in [("maxx", 1.0), ("maxy", 1.0), ("maxz", 1.0)] {
        state
            .heap_mut()
            .set_datum_field(world, field(name)?, Value::number(value))
            .map_err(|error| format!("failed to seed world.{name}: {error}"))?;
    }
    invoke(&mut executable, &procedures, GENESIS, world, &mut state)?;
    let config = match state.global(&field("config")?) {
        Some(Value::Datum(id)) => *id,
        value => return Err(format!("Genesis did not create config: {value:?}")),
    };
    invoke_with_args(
        &mut executable,
        &procedures,
        CONFIG_LOAD,
        config,
        &[Value::Null],
        &mut state,
    )?;
    let reagents = global_datum(&state, "SSreagents")?;
    invoke(&mut executable, &procedures, REAGENTS, reagents, &mut state)?;
    let id_access = global_datum(&state, "SSid_access")?;
    invoke(
        &mut executable,
        &procedures,
        ID_ACCESS,
        id_access,
        &mut state,
    )?;
    let dcs = global_datum(&state, "SSdcs")?;
    verify_neon_decal_smoothing(
        &mut executable,
        &procedures,
        &mut state,
        neon_carpet_fixture,
    )?;
    let knife_arguments = processable_arguments(&mut state, "knife")?;
    let saw_arguments = processable_arguments(&mut state, "saw")?;
    let knife_id = invoke_with_args(
        &mut executable,
        &procedures,
        DCS_GET_ID,
        dcs,
        &[Value::List(knife_arguments)],
        &mut state,
    )?;
    let saw_id = invoke_with_args(
        &mut executable,
        &procedures,
        DCS_GET_ID,
        dcs,
        &[Value::List(saw_arguments)],
        &mut state,
    )?;
    if knife_id == saw_id {
        return Err(format!(
            "processable knife/saw IDs collided: knife={knife_id:?} saw={saw_id:?}"
        ));
    }
    let knife_element = invoke_with_args(
        &mut executable,
        &procedures,
        DCS_GET_ELEMENT,
        dcs,
        &[Value::List(knife_arguments)],
        &mut state,
    )?;
    eprintln!(
        "early-assets-smoke: DCS after knife knife={:?} saw={:?}",
        dcs_cached_element(&state, dcs, &knife_id)?,
        dcs_cached_element(&state, dcs, &saw_id)?,
    );
    let saw_element = invoke_with_args(
        &mut executable,
        &procedures,
        DCS_GET_ELEMENT,
        dcs,
        &[Value::List(saw_arguments)],
        &mut state,
    )?;
    eprintln!(
        "early-assets-smoke: DCS after saw knife={:?} saw={:?}",
        dcs_cached_element(&state, dcs, &knife_id)?,
        dcs_cached_element(&state, dcs, &saw_id)?,
    );
    if knife_element == saw_element {
        return Err(format!(
            "processable knife/saw singletons collided: knife={knife_id:?} saw={saw_id:?} element={knife_element:?}"
        ));
    }
    let seed_field = field("seed")?;
    let seedless_growns = state
        .heap()
        .datums()
        .filter_map(|(_, datum)| {
            datum
                .type_path()
                .as_str()
                .starts_with("/obj/item/food/grown")
                .then(|| {
                    let seed = datum.field(&seed_field).ok().cloned();
                    (datum.type_path().to_string(), seed)
                })
        })
        .filter(|(_, seed)| {
            seed.as_ref()
                .is_none_or(|value| matches!(value, Value::Null))
        })
        .collect::<Vec<_>>();
    if !seedless_growns.is_empty() {
        eprintln!(
            "early-assets-smoke: expected diagnostic seedless grown prototypes={seedless_growns:?}"
        );
    }
    let greyscale = global_datum(&state, "SSgreyscale")?;
    invoke(
        &mut executable,
        &procedures,
        GREYSCALE,
        greyscale,
        &mut state,
    )?;
    drain_scheduler(&mut executable, &mut state)?;
    state
        .heap_mut()
        .set_datum_field(greyscale, field("initialized")?, Value::number(1.0))
        .map_err(|error| format!("failed to mark SSgreyscale initialized: {error}"))?;
    let early_assets = global_datum(&state, "SSearly_assets")?;
    eprintln!(
        "early-assets-smoke: invoking real SSearly_assets.Initialize elapsed_ms={}",
        started.elapsed().as_millis(),
    );
    invoke(
        &mut executable,
        &procedures,
        EARLY_ASSETS,
        early_assets,
        &mut state,
    )?;
    let research = global_datum(&state, "SSresearch")?;
    invoke(&mut executable, &procedures, RESEARCH, research, &mut state)?;
    for subsystem in ["SSfluids", "SSsmoke", "SSfoam"] {
        let fluid = global_datum(&state, subsystem)?;
        invoke(&mut executable, &procedures, FLUIDS, fluid, &mut state)?;
    }
    // Exercise the next full-boot frontier without paying for Mapping. A
    // production-sized but intentionally empty coordinate plane still runs
    // Monkestation's real canvas validation and complete holomap generation
    // path. In particular, the 480x480 DMI must not collapse to 32x32.
    for (name, value) in [("maxx", 255.0), ("maxy", 255.0)] {
        state
            .heap_mut()
            .set_datum_field(world, field(name)?, Value::number(value))
            .map_err(|error| format!("failed to seed world.{name}: {error}"))?;
    }
    let holomaps = global_datum(&state, "SSholomaps")?;
    eprintln!(
        "early-assets-smoke: invoking real SSholomaps.generate_holomap elapsed_ms={}",
        started.elapsed().as_millis(),
    );
    invoke_with_args(
        &mut executable,
        &procedures,
        GENERATE_HOLOMAP,
        holomaps,
        &[Value::number(1.0)],
        &mut state,
    )?;
    // Match Master's real order: Greyscale and all order-48..31 asset
    // producers run before SSatoms consumes the remaining world-wide
    // uninitialized atom queue.
    let atoms = global_datum(&state, "SSatoms")?;
    invoke_with_args(
        &mut executable,
        &procedures,
        BLOOD_PACK_INITIALIZE,
        blood_pack_fixture,
        &[Value::number(0.0), Value::Null],
        &mut state,
    )?;
    let reagents = match state
        .heap()
        .datum_field(blood_pack_fixture, &field("reagents")?)
        .map_err(|error| format!("blood pack did not retain its reagent holder: {error}"))?
    {
        Value::Datum(id) => *id,
        value => return Err(format!("blood-pack reagents is not a datum: {value:?}")),
    };
    if !matches!(
        state
            .heap()
            .datum_field(reagents, &field("reagent_list")?)
            .map_err(|error| format!("blood-pack reagent list is missing: {error}"))?,
        Value::List(_)
    ) {
        return Err("blood-pack oil reagent was not created".into());
    }
    invoke_with_args(
        &mut executable,
        &procedures,
        RUNE_SPAWN_INITIALIZE,
        rune_spawn_fixture,
        &[Value::number(0.0), Value::Null, Value::Null],
        &mut state,
    )?;
    if !matches!(
        state
            .heap()
            .datum_field(rune_spawn_fixture, &field("transform")?)
            .map_err(|error| format!("rune-spawn transform is missing: {error}"))?,
        Value::Datum(_)
    ) {
        return Err("rune-spawn matrix transform was not retained as a matrix datum".into());
    }
    invoke_with_args(
        &mut executable,
        &procedures,
        JUKEBOX_INITIALIZE,
        jukebox_fixture,
        &[Value::number(0.0)],
        &mut state,
    )?;
    let media_source = match state
        .heap()
        .datum_field(jukebox_fixture, &field("media_source")?)
        .map_err(|error| format!("jukebox did not retain its media source: {error}"))?
    {
        Value::Datum(id) => *id,
        value => return Err(format!("jukebox media_source is not a datum: {value:?}")),
    };
    let received_source = state
        .heap()
        .datum_field(media_source, &field("source")?)
        .map_err(|error| format!("jukebox media source did not bind source=: {error}"))?;
    if received_source != &Value::Datum(jukebox_fixture) {
        return Err(format!(
            "jukebox named constructor source bound incorrectly: {received_source:?}"
        ));
    }
    invoke_with_args(
        &mut executable,
        &procedures,
        KIRBY_INITIALIZE,
        kirby_fixture,
        &[Value::number(0.0)],
        &mut state,
    )?;
    let kirby_name = state
        .heap()
        .datum_field(kirby_fixture, &field("name")?)
        .map_err(|error| format!("Kirby plant fixture was not initialized: {error}"))?;
    if kirby_name != &Value::text("dead potted plant") {
        return Err(format!(
            "Kirby plant update_name did not execute the attached-colon ternary: {kirby_name:?}"
        ));
    }
    eprintln!(
        "early-assets-smoke: invoking real SSatoms.Initialize elapsed_ms={}",
        started.elapsed().as_millis(),
    );
    invoke(&mut executable, &procedures, ATOMS, atoms, &mut state)?;
    for (global, path) in [
        ("SSmachines", MACHINES),
        ("SSair", AIR),
        ("SSpersistence", PERSISTENCE),
        ("SSpersistent_paintings", PERSISTENT_PAINTINGS),
        ("SSassets", ASSETS),
        ("SSstation_coloring", STATION_COLORING),
        ("SSicon_smooth", ICON_SMOOTH),
        ("SSlighting", LIGHTING),
        ("SSshuttle", SHUTTLE),
        ("SScredits", CREDITS),
        ("SSinit_profiler", INIT_PROFILER),
    ] {
        let subsystem = global_datum(&state, global)?;
        eprintln!(
            "early-assets-smoke: invoking {global}.Initialize elapsed_ms={}",
            started.elapsed().as_millis(),
        );
        invoke(&mut executable, &procedures, path, subsystem, &mut state)?;
        drain_scheduler(&mut executable, &mut state)?;
    }
    eprintln!(
        "early-assets-smoke: PASS elapsed_ms={}",
        started.elapsed().as_millis()
    );
    Ok(())
}

fn effective_target(
    procedures: &ProcedureRegistry,
    path: &str,
) -> Result<ProcedureImplementationId, String> {
    procedures
        .procedures()
        .iter()
        .find(|procedure| procedure.path.to_string() == path)
        .and_then(|procedure| procedure.effective_target)
        .ok_or_else(|| format!("production procedure is missing: {path}"))
}

fn invoke(
    executable: &mut ExecutableProcedures,
    procedures: &ProcedureRegistry,
    path: &str,
    src: DatumId,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    invoke_with_args(executable, procedures, path, src, &[], state)
}

fn invoke_with_args(
    executable: &mut ExecutableProcedures,
    procedures: &ProcedureRegistry,
    path: &str,
    src: DatumId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let target = effective_target(procedures, path)?;
    let entry = executable
        .implementation(target)
        .ok_or_else(|| format!("linked VM entry is missing: {path}"))?;
    // Master.Initialize owns subsystem work as a scheduled continuation and
    // therefore slices each 10M-step quantum. This direct gate call has no
    // scheduler-owned outer frame, so allow several finite quanta while still
    // retaining a hard guard against a genuine infinite loop.
    execute_module_with_limits_in_context(
        executable.module(),
        entry,
        arguments,
        ExecutionLimits {
            max_steps: 100_000_000,
            ..ExecutionLimits::default()
        },
        state,
        &ExecutionContext::new(Value::Datum(src), Value::Null),
    )
    .map_err(|error| format!("{path} failed: {error}"))
}

fn field(name: &str) -> Result<FieldName, String> {
    FieldName::parse(name).map_err(|error| format!("invalid field {name:?}: {error}"))
}

fn global_datum(state: &ExecutionState, name: &str) -> Result<DatumId, String> {
    match state.global(&field(name)?) {
        Some(Value::Datum(id)) => Ok(*id),
        value => Err(format!("Genesis did not create {name}: {value:?}")),
    }
}

fn processable_arguments(
    state: &mut ExecutionState,
    tool: &str,
) -> Result<dm_value::ListId, String> {
    let list_id = state.heap_mut().allocate_list();
    let list = state
        .heap_mut()
        .list_mut(list_id)
        .map_err(|error| error.to_string())?;
    list.add(Value::TypePath(
        dm_value::TypePath::parse("/datum/element/processable")
            .map_err(|error| error.to_string())?,
    ));
    list.add(Value::text(tool));
    list.add(Value::TypePath(
        dm_value::TypePath::parse("/obj/item/food/breadslice/banana")
            .map_err(|error| error.to_string())?,
    ));
    list.add(Value::number(5.0));
    list.add(Value::number(30.0));
    list.set_key(Value::text("table_required"), Value::number(1.0));
    list.set_key(Value::text("screentip_verb"), Value::text("Slice"));
    Ok(list_id)
}

fn verify_neon_decal_smoothing(
    executable: &mut ExecutableProcedures,
    procedures: &ProcedureRegistry,
    state: &mut ExecutionState,
    neon_carpet: DatumId,
) -> Result<(), String> {
    let arguments = state.heap_mut().allocate_list();
    {
        let arguments = state
            .heap_mut()
            .list_mut(arguments)
            .map_err(|error| error.to_string())?;
        for value in [
            Value::TypePath(
                dm_value::TypePath::parse("/datum/element/decal")
                    .map_err(|error| error.to_string())?,
            ),
            Value::file("icons/turf/floors/carpet_neon_simple.dmi"),
            Value::text("light-nodots"),
            Value::number(2.0),
            Value::Null,
            Value::Null,
            Value::number(255.0),
            Value::text("#0000ff"),
            Value::number(255.0),
        ] {
            arguments.add(value);
        }
    }
    invoke_with_args(
        executable,
        procedures,
        ADD_ELEMENT,
        neon_carpet,
        &[Value::List(arguments)],
        state,
    )?;

    let decal_type =
        dm_value::TypePath::parse("/datum/element/decal").map_err(|error| error.to_string())?;
    let initial_decal = state
        .heap()
        .datums()
        .find_map(|(id, datum)| {
            (datum.type_path() == &decal_type
                && datum
                    .field(&field("smoothing").ok()?)
                    .ok()
                    .and_then(Value::as_number)
                    == Some(255.0))
            .then_some(id)
        })
        .ok_or("real decal Attach did not create the initial bespoke element")?;
    let initial_pic = match state
        .heap()
        .datum_field(initial_decal, &field("pic")?)
        .map_err(|error| error.to_string())?
    {
        Value::Datum(pic) => *pic,
        value => return Err(format!("initial decal pic is not an appearance: {value}")),
    };
    let initial_icon = state
        .heap()
        .datum_field(initial_pic, &field("icon")?)
        .map_err(|error| error.to_string())?;
    if initial_icon != &Value::file("icons/turf/floors/carpet_neon_simple.dmi") {
        return Err(format!(
            "typed mutable-appearance copy lost the decal icon before smoothing: {initial_icon}"
        ));
    }

    state
        .heap_mut()
        .set_datum_field(
            neon_carpet,
            field("smoothing_junction")?,
            Value::number(3.0),
        )
        .map_err(|error| error.to_string())?;
    invoke_with_args(
        executable,
        procedures,
        DECAL_SMOOTH_REACT,
        initial_decal,
        &[Value::Datum(neon_carpet)],
        state,
    )?;

    let replacement = state
        .heap()
        .datums()
        .find_map(|(id, datum)| {
            (datum.type_path() == &decal_type
                && datum
                    .field(&field("smoothing").ok()?)
                    .ok()
                    .and_then(Value::as_number)
                    == Some(3.0))
            .then_some(id)
        })
        .ok_or("smooth_react did not attach the replacement bespoke decal")?;
    let replacement_pic = match state
        .heap()
        .datum_field(replacement, &field("pic")?)
        .map_err(|error| error.to_string())?
    {
        Value::Datum(pic) => *pic,
        value => {
            return Err(format!(
                "replacement decal pic is not an appearance: {value}"
            ));
        }
    };
    let replacement_icon = state
        .heap()
        .datum_field(replacement_pic, &field("icon")?)
        .map_err(|error| error.to_string())?;
    if replacement_icon != &Value::file("icons/turf/floors/carpet_neon_simple.dmi") {
        return Err(format!(
            "smooth_react replacement lost the copied icon: {replacement_icon}"
        ));
    }
    eprintln!(
        "early-assets-smoke: real neon decal smooth_react passed initial={initial_decal:?} replacement={replacement:?}"
    );
    Ok(())
}

fn dcs_cached_element(
    state: &ExecutionState,
    dcs: DatumId,
    key: &Value,
) -> Result<Option<Value>, String> {
    let cache = state
        .heap()
        .datum_field(dcs, &field("elements_by_type")?)
        .map_err(|error| error.to_string())?;
    let Value::List(cache) = cache else {
        return Err(format!("SSdcs.elements_by_type is not a list: {cache:?}"));
    };
    Ok(state
        .heap()
        .list(*cache)
        .map_err(|error| error.to_string())?
        .get_key(key)
        .ok()
        .cloned())
}

fn drain_scheduler(
    executable: &mut ExecutableProcedures,
    state: &mut ExecutionState,
) -> Result<(), String> {
    let mut rounds = 0usize;
    while let Some(due) = state.next_scheduled_tick() {
        if rounds == 100_000 {
            return Err("scheduler did not quiesce while initializing prerequisites".to_owned());
        }
        let ticks = due.saturating_sub(state.scheduler_tick());
        advance_scheduler(
            executable.module(),
            ticks,
            ExecutionLimits::default(),
            state,
        )
        .map_err(|error| format!("prerequisite scheduler failed: {error}"))?;
        rounds += 1;
    }
    Ok(())
}

fn project_cache_file(environment: &Path) -> PathBuf {
    environment
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".dream64")
        .join("project-cache.bin")
}
