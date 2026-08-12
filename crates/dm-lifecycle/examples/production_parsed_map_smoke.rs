//! Fast production-project smoke test for Monkestation's runtime map reader.
//!
//! This intentionally compiles the supplied real `.dme` and invokes the real
//! `/datum/parsed_map` procedures with a one-cell in-memory TGM.  It avoids the
//! full Genesis/subsystem boot, while retaining the project's definitions,
//! inherited defaults, static regexes, and procedure dispatch graph.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::CompilerDatabase;
use dm_runtime::RuntimeImage;
use dm_semantics::{ExecutableProcedures, ProcedureImplementationId, ProcedureRegistry};
use dm_value::{DatumId, FieldName, TypePath, Value};
use dm_vm::{
    ExecutionContext, ExecutionLimits, ExecutionState, advance_scheduler, execute_module_in_context,
};

const PARSED_MAP: &str = "/datum/parsed_map";
const NEW: &str = "/datum/parsed_map/proc/New";
const BUILD_CACHE: &str = "/datum/parsed_map/proc/build_cache";
const TGM_LOAD: &str = "/datum/parsed_map/proc/_tgm_load";
const BUILD_COORDINATE: &str = "/datum/parsed_map/proc/build_coordinate";
const LOAD_MAP: &str = "/proc/load_map";

const ONE_CELL_TGM_PATH: &str =
    "crates/dm-lifecycle/examples/fixtures/production_parsed_map_smoke.dmm";

fn main() -> ExitCode {
    // Production procedures are macro-expanded deeply enough to exceed the
    // small default Windows main-thread stack while lowering.
    std::thread::Builder::new()
        .name("production-parsed-map-smoke".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .expect("failed to create production parsed-map smoke thread")
        .join()
        .unwrap_or_else(|_| Err("production parsed-map smoke thread panicked".to_owned()))
        .map_or_else(
            |error| {
                eprintln!("parsed-map-smoke: FAIL: {error}");
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
            .ok_or("usage: production_parsed_map_smoke <world.dme>")?,
    );
    if arguments.next().is_some() {
        return Err("usage: production_parsed_map_smoke <world.dme>".to_owned());
    }

    let started = Instant::now();
    eprintln!("parsed-map-smoke: compiling {}", environment.display());
    let cache = project_cache_file(&environment);
    let (compilation, cache_hit) = CompilerDatabase::new()
        .compile_cached(&environment, &cache)
        .map_err(|error| format!("project compilation failed: {error}"))?;
    eprintln!(
        "parsed-map-smoke: project ready elapsed_ms={} preprocessing_cache={}",
        started.elapsed().as_millis(),
        if cache_hit { "hit" } else { "miss" },
    );

    let procedures = ProcedureRegistry::build(&compilation);
    let roots = [NEW, BUILD_CACHE, TGM_LOAD, BUILD_COORDINATE, LOAD_MAP]
        .map(|path| effective_target(&procedures, path))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let mut executable = procedures
        .compile_vm_implementations_symbolic_dynamic(&compilation, roots)
        .map_err(|error| format!("parsed-map procedure lowering failed: {error}"))?;
    eprintln!(
        "parsed-map-smoke: linked deferred={}",
        executable.module().deferred_procedure_count(),
    );

    let mut runtime = RuntimeImage::from_compilation(&compilation)
        .map_err(|error| format!("runtime image construction failed: {error}"))?;
    let parsed_map = runtime
        .allocate_datum(&type_path(PARSED_MAP)?)
        .map_err(|error| format!("parsed-map allocation failed: {error}"))?;
    let glob = runtime
        .allocate_datum(&type_path("/datum/controller/global_vars")?)
        .map_err(|error| format!("GLOB allocation failed: {error}"))?;
    let master = runtime
        .allocate_datum(&type_path("/datum/controller/master")?)
        .map_err(|error| format!("Master allocation failed: {error}"))?;
    let ssatoms = runtime
        .allocate_datum(&type_path("/datum/controller/subsystem/atoms")?)
        .map_err(|error| format!("SSatoms allocation failed: {error}"))?;
    let mut state = runtime.take_execution_state();
    // Keep the production file procedures sandboxed, but point this bounded
    // smoke at the checked-in fixture rather than Monk's multi-megabyte maps.
    state.set_project_root(workspace_root());
    seed_minimal_globals(&mut state, glob, master, ssatoms)?;

    invoke(
        &mut executable,
        &procedures,
        NEW,
        parsed_map,
        &[Value::file(ONE_CELL_TGM_PATH)],
        &mut state,
    )?;
    let grid_key = parsed_grid_key(&state, parsed_map)?;
    if grid_key != Value::text("a") {
        return Err(format!(
            "New() retained the wrong one-cell grid key: {grid_key:?}"
        ));
    }
    eprintln!("parsed-map-smoke: New passed grid_key={grid_key:?}");

    let cache = invoke(
        &mut executable,
        &procedures,
        BUILD_CACHE,
        parsed_map,
        &[Value::number(0.0), Value::Null],
        &mut state,
    )?;
    let Value::List(cache_id) = cache else {
        return Err(format!("build_cache returned a non-list value: {cache:?}"));
    };
    let model = state
        .heap()
        .list(cache_id)
        .map_err(|error| format!("model cache is stale: {error}"))?
        .get_key(&Value::text("a"))
        .map_err(|error| format!("build_cache omitted model key \"a\": {error}"))?
        .clone();
    assert_model_text_macro(&state, &model)?;
    eprintln!("parsed-map-smoke: build_cache passed text_macro=\"Operative Remembrance Plaque\"");

    // A null coordinate is an intentional, supported early-return path in
    // build_coordinate. `_tgm_load` reaches that exact path because this
    // bounded harness does not materialize a second world geometry.
    invoke(
        &mut executable,
        &procedures,
        BUILD_COORDINATE,
        parsed_map,
        &[
            model,
            Value::Null,
            Value::number(0.0),
            Value::number(0.0),
            Value::number(0.0),
        ],
        &mut state,
    )?;
    eprintln!("parsed-map-smoke: build_coordinate passed");

    let infinity = Value::number(f32::INFINITY);
    let loaded = invoke(
        &mut executable,
        &procedures,
        TGM_LOAD,
        parsed_map,
        &[
            Value::number(1.0),
            Value::number(1.0),
            Value::number(1.0),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(f32::NEG_INFINITY),
            infinity.clone(),
            Value::number(f32::NEG_INFINITY),
            infinity.clone(),
            Value::number(f32::NEG_INFINITY),
            infinity,
            Value::number(0.0),
            Value::number(0.0),
        ],
        &mut state,
    )?;
    if loaded.as_number() != Some(1.0) {
        return Err(format!("_tgm_load did not report success: {loaded:?}"));
    }
    eprintln!("parsed-map-smoke: _tgm_load passed");

    // Exercise the real production call chain under cooperative scheduling:
    // load_map -> new /datum/parsed_map -> copy -> load -> _tgm_load. A high
    // initial tick usage and a zero Master limit guarantee that New's first
    // CHECK_TICK suspends the complete caller stack before map loading begins.
    set_numeric_datum_field(&mut state, master, "current_ticklimit", 0.0)?;
    set_numeric_datum_field(&mut state, master, "init_stage_completed", 0.0)?;
    set_world_numeric_field(&mut state, "tick_usage", 100.0)?;
    let scheduled_start = invoke_global(
        &mut executable,
        &procedures,
        LOAD_MAP,
        &[
            Value::file(ONE_CELL_TGM_PATH),
            Value::number(1.0),
            Value::number(1.0),
            Value::number(1.0),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(1.0),
            Value::number(f32::NEG_INFINITY),
            Value::number(f32::INFINITY),
            Value::number(f32::NEG_INFINITY),
            Value::number(f32::INFINITY),
            Value::number(f32::NEG_INFINITY),
            Value::number(f32::INFINITY),
            Value::number(0.0),
            Value::number(0.0),
        ],
        &mut state,
    )?;
    if scheduled_start != Value::Null || state.scheduled_task_count() == 0 {
        return Err(format!(
            "load_map did not cooperatively suspend: result={scheduled_start:?} pending={}",
            state.scheduled_task_count()
        ));
    }
    let (scheduled_results, scheduler_rounds) = drain_scheduler(executable.module(), &mut state)?;
    let scheduled_map = scheduled_results
        .into_iter()
        .find_map(|value| match value {
            Value::Datum(datum) => Some(datum),
            _ => None,
        })
        .ok_or("scheduled load_map completed without returning a parsed-map datum")?;
    let scheduled_grid_key = parsed_grid_key(&state, scheduled_map)?;
    if scheduled_grid_key != Value::text("a") {
        return Err(format!(
            "scheduled load_map retained the wrong one-cell grid key: {scheduled_grid_key:?}"
        ));
    }
    let bounds = state
        .heap()
        .datum_field(scheduled_map, &field("bounds")?)
        .map_err(|error| format!("scheduled _tgm_load omitted bounds: {error}"))?;
    if !matches!(bounds, Value::List(_)) {
        return Err(format!(
            "scheduled _tgm_load did not materialize bounds: {bounds:?}"
        ));
    }
    eprintln!(
        "parsed-map-smoke: scheduled load_map -> New -> _tgm_load passed rounds={scheduler_rounds} grid_key={scheduled_grid_key:?}"
    );
    eprintln!(
        "parsed-map-smoke: PASS elapsed_ms={}",
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
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    invoke_with_context(
        executable,
        procedures,
        path,
        arguments,
        state,
        &ExecutionContext::new(Value::Datum(src), Value::Null),
    )
}

fn invoke_global(
    executable: &mut ExecutableProcedures,
    procedures: &ProcedureRegistry,
    path: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    invoke_with_context(
        executable,
        procedures,
        path,
        arguments,
        state,
        &ExecutionContext::new(Value::Null, Value::Null),
    )
}

fn invoke_with_context(
    executable: &mut ExecutableProcedures,
    procedures: &ProcedureRegistry,
    path: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
    context: &ExecutionContext,
) -> Result<Value, String> {
    let target = effective_target(procedures, path)?;
    let entry = executable
        .implementation(target)
        .ok_or_else(|| format!("linked VM entry is missing: {path}"))?;
    execute_module_in_context(executable.module(), entry, arguments, state, context)
        .map_err(|error| format!("{path} failed: {error}"))
}

fn seed_minimal_globals(
    state: &mut ExecutionState,
    glob: DatumId,
    master: DatumId,
    ssatoms: DatumId,
) -> Result<(), String> {
    let empty = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(glob, field("map_model_default")?, Value::List(empty))
        .map_err(|error| format!("failed to seed GLOB.map_model_default: {error}"))?;
    let cached_maps = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(glob, field("cached_maps")?, Value::List(cached_maps))
        .map_err(|error| format!("failed to seed GLOB.cached_maps: {error}"))?;
    state.set_global(field("GLOB")?, Value::Datum(glob));
    state.set_global(field("Master")?, Value::Datum(master));
    state.set_global(field("SSatoms")?, Value::Datum(ssatoms));
    let world = match state.global(&field("world")?) {
        Some(Value::Datum(world)) => *world,
        value => return Err(format!("canonical world global is unavailable: {value:?}")),
    };
    for name in ["maxx", "maxy", "maxz"] {
        state
            .heap_mut()
            .set_datum_field(world, field(name)?, Value::number(1.0))
            .map_err(|error| format!("failed to seed world.{name}: {error}"))?;
    }
    set_world_numeric_field(state, "tick_lag", 1.0)?;
    set_world_numeric_field(state, "tick_usage", 0.0)?;
    Ok(())
}

fn set_numeric_datum_field(
    state: &mut ExecutionState,
    datum: DatumId,
    name: &str,
    value: f32,
) -> Result<(), String> {
    state
        .heap_mut()
        .set_datum_field(datum, field(name)?, Value::number(value))
        .map(|_| ())
        .map_err(|error| format!("failed to seed datum.{name}: {error}"))
}

fn set_world_numeric_field(
    state: &mut ExecutionState,
    name: &str,
    value: f32,
) -> Result<(), String> {
    let world = match state.global(&field("world")?) {
        Some(Value::Datum(world)) => *world,
        current => {
            return Err(format!(
                "canonical world global is unavailable: {current:?}"
            ));
        }
    };
    set_numeric_datum_field(state, world, name, value)
}

fn drain_scheduler(
    module: &dm_vm::Module,
    state: &mut ExecutionState,
) -> Result<(Vec<Value>, usize), String> {
    let mut completed = Vec::new();
    let mut rounds = 0_usize;
    while state.scheduled_task_count() != 0 {
        if rounds >= 1_000 {
            return Err(format!(
                "scheduled map load did not quiesce after {rounds} rounds (pending={})",
                state.scheduled_task_count()
            ));
        }
        let due = state
            .next_scheduled_tick()
            .ok_or("scheduled task count was nonzero without a due tick")?;
        let ticks = due.saturating_sub(state.scheduler_tick());
        completed.extend(
            advance_scheduler(module, ticks, ExecutionLimits::default(), state)
                .map_err(|error| format!("scheduled production map load failed: {error}"))?,
        );
        rounds += 1;
    }
    Ok((completed, rounds))
}

fn parsed_grid_key(state: &ExecutionState, parsed_map: DatumId) -> Result<Value, String> {
    let grid_sets = match state
        .heap()
        .datum_field(parsed_map, &field("gridSets")?)
        .map_err(|error| format!("New() did not materialize gridSets: {error}"))?
    {
        Value::List(list) => *list,
        value => return Err(format!("gridSets is not a list: {value:?}")),
    };
    let first_grid = match state
        .heap()
        .list(grid_sets)
        .map_err(|error| format!("gridSets is stale: {error}"))?
        .get(1)
        .map_err(|error| format!("New() produced no grid set: {error}"))?
    {
        Value::Datum(datum) => *datum,
        value => return Err(format!("gridSets[1] is not a datum: {value:?}")),
    };
    let grid_lines = match state
        .heap()
        .datum_field(first_grid, &field("gridLines")?)
        .map_err(|error| format!("grid set omitted gridLines: {error}"))?
    {
        Value::List(list) => *list,
        value => return Err(format!("gridLines is not a list: {value:?}")),
    };
    state
        .heap()
        .list(grid_lines)
        .map_err(|error| format!("gridLines is stale: {error}"))?
        .get(1)
        .cloned()
        .map_err(|error| format!("gridLines is empty: {error}"))
}

fn assert_model_text_macro(state: &ExecutionState, model: &Value) -> Result<(), String> {
    let Value::List(model) = model else {
        return Err(format!("cached model is not a list: {model:?}"));
    };
    let member_attributes = match state
        .heap()
        .list(*model)
        .map_err(|error| format!("cached model is stale: {error}"))?
        .get(2)
        .map_err(|error| format!("cached model omitted member attributes: {error}"))?
    {
        Value::List(list) => *list,
        value => return Err(format!("cached member attributes is not a list: {value:?}")),
    };
    let plaque_attributes = match state
        .heap()
        .list(member_attributes)
        .map_err(|error| format!("cached member attributes is stale: {error}"))?
        .get(1)
        .map_err(|error| format!("cached plaque attributes are missing: {error}"))?
    {
        Value::List(list) => *list,
        value => return Err(format!("cached plaque attributes is not a list: {value:?}")),
    };
    let name = state
        .heap()
        .list(plaque_attributes)
        .map_err(|error| format!("cached plaque attributes are stale: {error}"))?
        .get_key(&Value::text("name"))
        .map_err(|error| format!("cached plaque name is missing: {error}"))?;
    if name != &Value::text("Operative Remembrance Plaque") {
        return Err(format!(
            "apply_text_macros retained the wrong plaque name: {name:?}"
        ));
    }
    Ok(())
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
    // Examples live one directory deeper than the main binary
    // (`target/{profile}/examples`). Keep them on the same production cache
    // instead of accidentally creating `target/debug/dream64-cache`.
    let cache_root = env::var_os("DREAM64_CACHE_DIR").map_or_else(
        || PathBuf::from("target").join("dream64-cache"),
        PathBuf::from,
    );
    cache_root.join(format!("project-{hash:016x}.bin"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("dm-lifecycle must live below the workspace root")
        .to_path_buf()
}
