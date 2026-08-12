//! Bounded production-project smoke test for Monkestation's map-template preload.
//!
//! This compiles the real project and invokes the real mapping subsystem's
//! `preloadTemplates()` without running Genesis or allocating world geometry.
//! It covers `flist("_maps/templates/")`, the directory entry
//! `lazy_templates/`, and the chained ruin/shuttle/shelter/holodeck/template
//! constructors that parse their map files during preload.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::CompilerDatabase;
use dm_runtime::RuntimeImage;
use dm_semantics::{ExecutableProcedures, ProcedureImplementationId, ProcedureRegistry};
use dm_value::{DatumId, FieldName, TypePath, Value};
use dm_vm::{ExecutionContext, ExecutionState, execute_module_in_context};

const PRELOAD_TEMPLATES: &str = "/datum/controller/subsystem/mapping/proc/preloadTemplates";
const INIT_FILENAME_FILTER: &str =
    "/datum/controller/global_vars/proc/InitGlobalfilename_forbidden_chars";

fn main() -> ExitCode {
    std::thread::Builder::new()
        .name("production-preload-templates-smoke".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .expect("failed to create production preload-templates smoke thread")
        .join()
        .unwrap_or_else(|_| Err("production preload-templates smoke thread panicked".to_owned()))
        .map_or_else(
            |error| {
                eprintln!("preload-templates-smoke: FAIL: {error}");
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
            .ok_or("usage: production_preload_templates_smoke <world.dme>")?,
    );
    if arguments.next().is_some() {
        return Err("usage: production_preload_templates_smoke <world.dme>".to_owned());
    }
    let project_root = environment
        .parent()
        .ok_or_else(|| format!("project has no parent directory: {}", environment.display()))?
        .to_path_buf();

    let started = Instant::now();
    eprintln!(
        "preload-templates-smoke: compiling {}",
        environment.display()
    );
    let cache = project_cache_file(&environment);
    let (compilation, cache_hit) = CompilerDatabase::new()
        .compile_cached(&environment, &cache)
        .map_err(|error| format!("project compilation failed: {error}"))?;
    eprintln!(
        "preload-templates-smoke: project ready elapsed_ms={} preprocessing_cache={}",
        started.elapsed().as_millis(),
        if cache_hit { "hit" } else { "miss" },
    );

    let procedures = ProcedureRegistry::build(&compilation);
    let roots = [PRELOAD_TEMPLATES, INIT_FILENAME_FILTER]
        .map(|path| effective_target(&procedures, path))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let mut executable = procedures
        .compile_vm_implementations_symbolic_dynamic(&compilation, roots)
        .map_err(|error| format!("preloadTemplates lowering failed: {error}"))?;
    eprintln!(
        "preload-templates-smoke: linked deferred={}",
        executable.module().deferred_procedure_count(),
    );

    let mut runtime = RuntimeImage::from_compilation(&compilation)
        .map_err(|error| format!("runtime image construction failed: {error}"))?;
    let mapping = allocate(
        &mut runtime,
        "/datum/controller/subsystem/mapping",
        "SSmapping",
    )?;
    let current_map = allocate(&mut runtime, "/datum/map_config", "current map")?;
    let glob = allocate(&mut runtime, "/datum/controller/global_vars", "GLOB")?;
    let master = allocate(&mut runtime, "/datum/controller/master", "Master")?;
    let config = allocate(&mut runtime, "/datum/controller/configuration", "config")?;
    let mut state = runtime.take_execution_state();
    state.set_project_root(project_root.clone());
    seed_minimal_globals(&mut state, mapping, current_map, glob, master, config)?;
    invoke(
        &mut executable,
        &procedures,
        INIT_FILENAME_FILTER,
        glob,
        &mut state,
    )?;

    let root_entries = template_root_entries(&project_root)?;
    eprintln!(
        "preload-templates-smoke: invoking real preloadTemplates root_entries={} first={:?}",
        root_entries.len(),
        root_entries.first(),
    );
    invoke(
        &mut executable,
        &procedures,
        PRELOAD_TEMPLATES,
        mapping,
        &mut state,
    )?;

    let templates = list_field(&state, mapping, "map_templates")?;
    let loaded = state
        .heap()
        .list(templates)
        .map_err(|error| format!("SSmapping.map_templates is stale: {error}"))?;
    for entry in &root_entries {
        let value = loaded
            .get_key(&Value::text(entry.as_str()))
            .map_err(|error| format!("root template {entry:?} was not registered: {error}"))?;
        if !matches!(value, Value::Datum(_)) {
            return Err(format!(
                "root template {entry:?} registered a non-datum value: {value:?}"
            ));
        }
    }
    let lazy = loaded
        .get_key(&Value::text("lazy_templates/"))
        .map_err(|error| format!("lazy_templates/ directory entry was not registered: {error}"))?;
    if !matches!(lazy, Value::Datum(_)) {
        return Err(format!(
            "lazy_templates/ registered a non-datum value: {lazy:?}"
        ));
    }

    eprintln!(
        "preload-templates-smoke: PASS elapsed_ms={} root_entries={} registered_templates={} lazy_templates=nonfatal",
        started.elapsed().as_millis(),
        root_entries.len(),
        loaded.len(),
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
    let target = effective_target(procedures, path)?;
    let entry = executable
        .implementation(target)
        .ok_or_else(|| format!("linked VM entry is missing: {path}"))?;
    execute_module_in_context(
        executable.module(),
        entry,
        &[],
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

fn seed_minimal_globals(
    state: &mut ExecutionState,
    mapping: DatumId,
    current_map: DatumId,
    glob: DatumId,
    master: DatumId,
    config: DatumId,
) -> Result<(), String> {
    set_field(state, mapping, "current_map", Value::Datum(current_map))?;
    for name in ["map_model_default", "cached_maps", "ruin_config"] {
        let list = state.heap_mut().allocate_list();
        let value = Value::List(list);
        set_field(state, glob, name, value.clone())?;
        state.set_global(field(name)?, value);
    }
    state.set_global(field("SSmapping")?, Value::Datum(mapping));
    state.set_global(field("GLOB")?, Value::Datum(glob));
    state.set_global(field("Master")?, Value::Datum(master));
    state.set_global(field("config")?, Value::Datum(config));
    Ok(())
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

fn list_field(
    state: &ExecutionState,
    datum: DatumId,
    name: &str,
) -> Result<dm_value::ListId, String> {
    match state
        .heap()
        .datum_field(datum, &field(name)?)
        .map_err(|error| format!("datum.{name} is unavailable: {error}"))?
    {
        Value::List(list) => Ok(*list),
        value => Err(format!("datum.{name} is not a list: {value:?}")),
    }
}

fn template_root_entries(project_root: &Path) -> Result<Vec<String>, String> {
    let directory = project_root.join("_maps").join("templates");
    let mut entries = std::fs::read_dir(&directory)
        .map_err(|error| format!("cannot enumerate {}: {error}", directory.display()))?
        .map(|entry| {
            let entry = entry.map_err(|error| error.to_string())?;
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if entry
                .file_type()
                .map_err(|error| error.to_string())?
                .is_dir()
            {
                name.push('/');
            }
            Ok(name)
        })
        .collect::<Result<Vec<_>, String>>()?;
    entries.sort();
    Ok(entries)
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
