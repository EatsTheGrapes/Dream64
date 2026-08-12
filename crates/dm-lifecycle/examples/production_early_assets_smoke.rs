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
const GREYSCALE: &str = "/datum/controller/subsystem/processing/greyscale/proc/Initialize";
const EARLY_ASSETS: &str = "/datum/controller/subsystem/early_assets/proc/Initialize";
const GENERATE_HOLOMAP: &str = "/datum/controller/subsystem/holomaps/proc/generate_holomap";

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
        GREYSCALE,
        EARLY_ASSETS,
        GENERATE_HOLOMAP,
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
    let world = runtime
        .canonical_world()
        .ok_or("canonical /world datum is missing")?;
    let mut state = runtime.take_execution_state();
    state.set_project_root(project_root);
    state.set_global(field("world")?, Value::Datum(world));
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
