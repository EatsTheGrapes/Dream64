//! Bounded production-project smoke test for Monkestation's Lavaland terrain generator.
//!
//! This compiles the real project and invokes the inherited production
//! `/datum/map_generator/cave_generator/generate_terrain()` implementation on
//! a real `/datum/map_generator/cave_generator/lavaland` datum.  A tiny 4x4
//! synthetic turf set keeps the gate independent of Genesis and full world
//! allocation while still exercising rust-g `cnoise_generate`, BYOND text
//! indexing, open/closed selection, and dynamic turf replacement.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::CompilerDatabase;
use dm_runtime::RuntimeImage;
use dm_semantics::{ExecutableProcedures, ProcedureImplementationId, ProcedureRegistry};
use dm_value::{DatumId, FieldName, ListId, TypePath, Value};
use dm_vm::{ExecutionContext, ExecutionState, execute_module_in_context};

const GENERATE_TERRAIN: &str = "/datum/map_generator/cave_generator/proc/generate_terrain";
const GENERATOR: &str = "/datum/map_generator/cave_generator/lavaland";
const GENERATION_AREA: &str = "/area/lavaland/surface/outdoors/unexplored";
const INPUT_TURF: &str = "/turf/open/space";
const OPEN_TURF: &str = "/turf/open/misc/asteroid/basalt/lava_land_surface";
const CLOSED_TURF: &str = "/turf/closed/mineral/random/volcanic";
const WIDTH: usize = 4;
const HEIGHT: usize = 4;

fn main() -> ExitCode {
    std::thread::Builder::new()
        .name("production-lavaland-terrain-smoke".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .expect("failed to create production Lavaland terrain smoke thread")
        .join()
        .unwrap_or_else(|_| Err("production Lavaland terrain smoke thread panicked".to_owned()))
        .map_or_else(
            |error| {
                eprintln!("lavaland-terrain-smoke: FAIL: {error}");
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
            .ok_or("usage: production_lavaland_terrain_smoke <world.dme>")?,
    );
    if arguments.next().is_some() {
        return Err("usage: production_lavaland_terrain_smoke <world.dme>".to_owned());
    }

    let started = Instant::now();
    eprintln!(
        "lavaland-terrain-smoke: compiling {}",
        environment.display()
    );
    let cache = project_cache_file(&environment);
    let (compilation, cache_hit) = CompilerDatabase::new()
        .compile_cached(&environment, &cache)
        .map_err(|error| format!("project compilation failed: {error}"))?;
    eprintln!(
        "lavaland-terrain-smoke: project ready elapsed_ms={} preprocessing_cache={}",
        started.elapsed().as_millis(),
        if cache_hit { "hit" } else { "miss" },
    );

    let procedures = ProcedureRegistry::build(&compilation);
    let root = effective_target(&procedures, GENERATE_TERRAIN)?;
    let mut executable = procedures
        .compile_vm_implementations_symbolic_dynamic(&compilation, [root])
        .map_err(|error| format!("Lavaland terrain lowering failed: {error}"))?;
    eprintln!(
        "lavaland-terrain-smoke: linked deferred={}",
        executable.module().deferred_procedure_count(),
    );

    let mut runtime = RuntimeImage::from_compilation(&compilation)
        .map_err(|error| format!("runtime image construction failed: {error}"))?;
    let generator = allocate(&mut runtime, GENERATOR, "Lavaland generator")?;
    let area = allocate(&mut runtime, GENERATION_AREA, "Lavaland area")?;
    let title = allocate(&mut runtime, "/datum/controller/subsystem/title", "SStitle")?;
    let ticker = allocate(
        &mut runtime,
        "/datum/controller/subsystem/ticker",
        "SSticker",
    )?;
    let glob = allocate(&mut runtime, "/datum/controller/global_vars", "GLOB")?;
    let input_turfs = (0..WIDTH * HEIGHT)
        .map(|index| {
            allocate(
                &mut runtime,
                INPUT_TURF,
                &format!("input turf {}", index + 1),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut state = runtime.take_execution_state();
    let project_root = environment
        .parent()
        .ok_or_else(|| format!("project has no parent directory: {}", environment.display()))?;
    state.set_project_root(project_root.to_path_buf());
    seed_harness(
        &mut state,
        generator,
        area,
        title,
        ticker,
        glob,
        &input_turfs,
    )?;
    // Production `RunTerrainGeneration()` snapshots area.contents into a
    // separate list before replacements mutate the area's live contents.
    // Keep the same aliasing contract here so this smoke exercises the real
    // generator loop rather than a mutation-during-iteration artifact.
    let generation_turfs = value_list(&mut state, input_turfs.iter().copied().map(Value::Datum))?;

    let open_before = count_exact_type(&state, OPEN_TURF)?;
    let closed_before = count_exact_type(&state, CLOSED_TURF)?;
    eprintln!(
        "lavaland-terrain-smoke: invoking real generator type={GENERATOR} dimensions={WIDTH}x{HEIGHT} defaults=45/50/4/3",
    );
    invoke(
        &mut executable,
        &procedures,
        GENERATE_TERRAIN,
        generator,
        &[Value::List(generation_turfs), Value::Datum(area)],
        &mut state,
    )?;

    let generated = text_field(&state, generator, "string_gen")?;
    let expected_len = WIDTH * HEIGHT;
    if generated.len() != expected_len {
        return Err(format!(
            "cnoise_generate returned {} bytes for a {WIDTH}x{HEIGHT} grid; expected {expected_len}",
            generated.len()
        ));
    }
    if let Some(invalid) = generated.bytes().find(|byte| !matches!(byte, b'0' | b'1')) {
        return Err(format!(
            "cnoise_generate returned non-binary byte {invalid:#x}: {generated:?}"
        ));
    }

    let expected_open = generated.bytes().filter(|byte| *byte == b'0').count();
    let expected_closed = expected_len - expected_open;
    let created_open = count_exact_type(&state, OPEN_TURF)? - open_before;
    let created_closed = count_exact_type(&state, CLOSED_TURF)? - closed_before;
    if created_open != expected_open || created_closed != expected_closed {
        return Err(format!(
            "terrain did not consume cnoise cells correctly: output={generated:?} expected_open={expected_open} expected_closed={expected_closed} created_open={created_open} created_closed={created_closed}"
        ));
    }
    if created_open + created_closed != input_turfs.len() {
        return Err(format!(
            "terrain replaced {} of {} supplied turfs",
            created_open + created_closed,
            input_turfs.len()
        ));
    }

    eprintln!(
        "lavaland-terrain-smoke: PASS elapsed_ms={} cnoise_bytes={} open={} closed={} replaced={}",
        started.elapsed().as_millis(),
        generated.len(),
        created_open,
        created_closed,
        created_open + created_closed,
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
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let target = effective_target(procedures, path)?;
    let entry = executable
        .implementation(target)
        .ok_or_else(|| format!("linked VM entry is missing: {path}"))?;
    execute_module_in_context(
        executable.module(),
        entry,
        arguments,
        state,
        &ExecutionContext::new(Value::Datum(src), Value::Null),
    )
    .map_err(|error| format!("{path} failed: {error}"))
}

fn allocate(runtime: &mut RuntimeImage, path: &str, label: &str) -> Result<DatumId, String> {
    runtime
        .allocate_datum(&type_path(path)?)
        .map_err(|error| format!("{label} allocation failed: {error}"))
}

#[allow(clippy::too_many_arguments)]
fn seed_harness(
    state: &mut ExecutionState,
    generator: DatumId,
    area: DatumId,
    title: DatumId,
    ticker: DatumId,
    glob: DatumId,
    turfs: &[DatumId],
) -> Result<(), String> {
    state.set_global(field("SStitle")?, Value::Datum(title));
    state.set_global(field("SSticker")?, Value::Datum(ticker));
    state.set_global(field("GLOB")?, Value::Datum(glob));

    set_world_number(state, "maxx", WIDTH as f32)?;
    set_world_number(state, "maxy", HEIGHT as f32)?;
    set_world_number(state, "maxz", 1.0)?;
    set_world_number(state, "tick_lag", 1.0)?;
    set_world_number(state, "tick_usage", 0.0)?;

    let open_types = singleton_list(state, Value::TypePath(type_path(OPEN_TURF)?))?;
    let closed_types = singleton_list(state, Value::TypePath(type_path(CLOSED_TURF)?))?;
    set_field(state, generator, "open_turf_types", Value::List(open_types))?;
    set_field(
        state,
        generator,
        "closed_turf_types",
        Value::List(closed_types),
    )?;

    let area_contents = list_field(state, area, "contents")?;
    let world = world_datum(state)?;
    let world_contents = list_field(state, world, "contents")?;
    for (index, turf) in turfs.iter().copied().enumerate() {
        let x = index % WIDTH + 1;
        let y = index / WIDTH + 1;
        set_field(state, turf, "x", Value::number(x as f32))?;
        set_field(state, turf, "y", Value::number(y as f32))?;
        set_field(state, turf, "z", Value::number(1.0))?;
        set_field(state, turf, "loc", Value::Datum(area))?;
        state
            .heap_mut()
            .list_mut(area_contents)
            .map_err(|error| format!("Lavaland area.contents is stale: {error}"))?
            .add(Value::Datum(turf));
        state
            .heap_mut()
            .list_mut(world_contents)
            .map_err(|error| format!("world.contents is stale: {error}"))?
            .add(Value::Datum(turf));
    }
    Ok(())
}

fn singleton_list(state: &mut ExecutionState, value: Value) -> Result<ListId, String> {
    value_list(state, [value])
}

fn value_list(
    state: &mut ExecutionState,
    values: impl IntoIterator<Item = Value>,
) -> Result<ListId, String> {
    let list = state.heap_mut().allocate_list();
    let list_value = state
        .heap_mut()
        .list_mut(list)
        .map_err(|error| format!("new list is stale: {error}"))?;
    for value in values {
        list_value.add(value);
    }
    Ok(list)
}

fn set_world_number(state: &mut ExecutionState, name: &str, value: f32) -> Result<(), String> {
    let world = world_datum(state)?;
    set_field(state, world, name, Value::number(value))
}

fn world_datum(state: &ExecutionState) -> Result<DatumId, String> {
    match state.global(&field("world")?) {
        Some(Value::Datum(world)) => Ok(*world),
        value => Err(format!("world global is not a datum: {value:?}")),
    }
}

fn set_field(
    state: &mut ExecutionState,
    datum: DatumId,
    name: &str,
    value: Value,
) -> Result<(), String> {
    state
        .heap_mut()
        .set_datum_field(datum, field(name)?, value)
        .map(|_| ())
        .map_err(|error| format!("failed to seed datum.{name}: {error}"))
}

fn list_field(state: &ExecutionState, datum: DatumId, name: &str) -> Result<ListId, String> {
    match state
        .heap()
        .datum_field(datum, &field(name)?)
        .map_err(|error| format!("datum.{name} is unavailable: {error}"))?
    {
        Value::List(list) => Ok(*list),
        value => Err(format!("datum.{name} is not a list: {value:?}")),
    }
}

fn text_field(state: &ExecutionState, datum: DatumId, name: &str) -> Result<String, String> {
    match state
        .heap()
        .datum_field(datum, &field(name)?)
        .map_err(|error| format!("datum.{name} is unavailable: {error}"))?
    {
        Value::Text(text) => Ok(text.to_string()),
        value => Err(format!("datum.{name} is not text: {value:?}")),
    }
}

fn count_exact_type(state: &ExecutionState, path: &str) -> Result<usize, String> {
    let path = type_path(path)?;
    Ok(state
        .heap()
        .datums()
        .filter(|(_, datum)| datum.type_path() == &path)
        .count())
}

fn field(name: &str) -> Result<FieldName, String> {
    FieldName::parse(name).map_err(|error| error.to_string())
}

fn type_path(path: &str) -> Result<TypePath, String> {
    TypePath::parse(path).map_err(|error| error.to_string())
}

fn project_cache_file(environment: &Path) -> PathBuf {
    let canonical =
        std::fs::canonicalize(environment).unwrap_or_else(|_| environment.to_path_buf());
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let cache_root = env::var_os("DREAM64_CACHE_DIR").map_or_else(
        || PathBuf::from("target").join("dream64-cache"),
        PathBuf::from,
    );
    cache_root.join(format!("project-{hash:016x}.bin"))
}
