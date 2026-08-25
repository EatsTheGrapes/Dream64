use std::collections::BTreeMap;
use std::collections::hash_map::RandomState;
use std::env;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead as _, BufReader, BufWriter, Read as _, Write as _};
use std::net::{IpAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dm_compiler::persistent_database::{
    PersistentCompilerDatabase, decode_stable_id_table, encode_stable_id_table,
};
use dm_compiler::{
    BuildMode, Compilation, CompilerDatabase, PERSISTENT_EXECUTABLE_PAGE_BASE,
    PERSISTENT_EXECUTABLE_SECTION,
};
use dm_lifecycle::ipc::{BootPhase, LoopbackIpc, parse_loopback_address};
use dm_lifecycle::{
    HeadlessReadinessProbe, HostSliceBudget, LifecycleIndex, LifecycleKind, LifecycleResolution,
    SchedulerDrainLimits, SchedulerDrainTermination, advance_persistent_scheduler_responsive,
    artifact::{ArtifactSection, CompiledArtifact, engine_semantics_fingerprint},
    audit_initialization_plan_with_precompiled, build_initialization_plan, build_parsed_dmm_cache,
    decode_and_attach_procedure_semantics, decode_dmm_measurements, decode_parsed_dmm_cache,
    dmm_measurements_from_parsed, encode_dmm_measurements, encode_parsed_dmm_cache,
    encode_procedure_semantics, execute_boot_initialization_plan_with_precompiled,
    execute_boot_initialization_plan_with_precompiled_and_startup_service,
    precompile_lifecycle_for_world, precompile_lifecycle_for_world_with_executable,
    precompile_portable_lifecycle_for_world, readiness_probe_matches,
    sweep_lifecycle_compatibility, sweep_lifecycle_compatibility_with_closures,
};
use dm_project::Project;
use dm_runtime::{
    RuntimeImage, RuntimeImageConstructionEvent, RuntimeInitializerDiagnostic,
    RuntimeStructuralSeed,
};
use dm_semantics::{ExecutableProcedures, ProcedureRegistry};
use dm_value::{FieldName, TypePath, Value};
use dm_vm::{ExecutionLimits, advance_scheduler};
use dm_world::allocate_world;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

const MAX_EAGER_ARTIFACT_DIAGNOSTICS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Command {
    Compile,
    Plan,
    Boot,
    Sweep,
    SweepClosure,
    LobbyPreflight,
    LobbyPreview,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProductionReadyWorldIdentity {
    random_seed: u64,
    deployment_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ReadyWorldMode {
    Disabled,
    Development,
    Prewarm(ProductionReadyWorldIdentity),
    Activate(ProductionReadyWorldIdentity),
}

impl ReadyWorldMode {
    const fn production_identity(&self) -> Option<&ProductionReadyWorldIdentity> {
        match self {
            Self::Prewarm(identity) | Self::Activate(identity) => Some(identity),
            Self::Disabled | Self::Development => None,
        }
    }

    const fn writes_snapshot(&self) -> bool {
        matches!(self, Self::Development | Self::Prewarm(_))
    }
}

fn parse_ready_world_mode(
    prewarm: bool,
    activate: bool,
    development: bool,
    disabled: bool,
    random_seed: Option<&str>,
    deployment_id: Option<&str>,
) -> Result<ReadyWorldMode, String> {
    if disabled {
        return Ok(ReadyWorldMode::Disabled);
    }
    if prewarm && activate {
        return Err(
            "DREAM64_PREWARM_READY_WORLD and DREAM64_ACTIVATE_READY_WORLD are mutually exclusive"
                .to_owned(),
        );
    }
    if !prewarm && !activate {
        return Ok(if development {
            ReadyWorldMode::Development
        } else {
            ReadyWorldMode::Disabled
        });
    }
    let random_seed = random_seed
        .ok_or_else(|| "production ready-world mode requires DREAM64_RANDOM_SEED".to_owned())?
        .parse::<u64>()
        .map_err(|_| "DREAM64_RANDOM_SEED must be a nonzero u64".to_owned())?;
    if random_seed == 0 {
        return Err("DREAM64_RANDOM_SEED must be a nonzero u64".to_owned());
    }
    let deployment_id = deployment_id
        .map(str::trim)
        .filter(|identity| !identity.is_empty())
        .ok_or_else(|| "production ready-world mode requires DREAM64_DEPLOYMENT_ID".to_owned())?
        .to_owned();
    let identity = ProductionReadyWorldIdentity {
        random_seed,
        deployment_id,
    };
    Ok(if prewarm {
        ReadyWorldMode::Prewarm(identity)
    } else {
        ReadyWorldMode::Activate(identity)
    })
}

fn ready_world_mode_from_environment() -> Result<ReadyWorldMode, String> {
    let enabled = |name| env::var(name).is_ok_and(|value| value.trim() == "1");
    parse_ready_world_mode(
        enabled("DREAM64_PREWARM_READY_WORLD"),
        enabled("DREAM64_ACTIVATE_READY_WORLD"),
        env::var_os("DREAM64_ENABLE_READY_WORLD_CACHE").is_some(),
        env::var_os("DREAM64_DISABLE_READY_CACHE").is_some(),
        env::var("DREAM64_RANDOM_SEED").ok().as_deref(),
        env::var("DREAM64_DEPLOYMENT_ID").ok().as_deref(),
    )
}

const fn progress_label(command: Command) -> &'static str {
    match command {
        Command::Compile => "compile-progress",
        Command::LobbyPreflight => "lobby-preflight-progress",
        Command::LobbyPreview => "lobby-preview-progress",
        _ => "boot-progress",
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> ExitCode {
    // Large production DM procedures (notably tg/Monk's macro-expanded
    // /atom/Initialize) legitimately create a deeply nested compiler walk.
    // Windows' default main-thread stack is too small for that workload.
    // Keep the whole persistent headless host on one explicitly sized stack
    // so deferred materialization and later runtime calls have the same
    // predictable capacity.
    std::thread::Builder::new()
        .name("dream64-headless".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(run_main)
        .expect("failed to create Dream64 headless host thread")
        .join()
        .unwrap_or_else(|_| {
            eprintln!("Dream64 headless host thread panicked");
            ExitCode::FAILURE
        })
}

fn report_public_endpoint(port: u16) {
    let _ = std::thread::Builder::new()
        .name("dream64-public-ip".to_owned())
        .spawn(move || {
            let output = ProcessCommand::new("curl")
                .args([
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    "5",
                    "https://api.ipify.org",
                ])
                .output();
            let Ok(output) = output else {
                eprintln!("server-network: public IP discovery unavailable");
                return;
            };
            let address = String::from_utf8_lossy(&output.stdout);
            match address.trim().parse::<IpAddr>() {
                Ok(address) => eprintln!(
                    "server-network: public-endpoint={address}:{port} tcp-port-forward-required=true"
                ),
                Err(_) => eprintln!("server-network: public IP discovery unavailable"),
            }
        });
}

#[allow(clippy::too_many_lines)]
fn run_main() -> ExitCode {
    let process_started = Instant::now();
    let mut arguments = env::args_os().skip(1);
    let Some(first) = arguments.next() else {
        eprintln!(
            "usage: dream64-server [plan|boot|sweep|sweep-closure|lobby-preflight|lobby-preview] <world.dme> [map.dmm]"
        );
        return ExitCode::from(2);
    };
    let (command, environment) = if first.as_os_str() == OsStr::new("compile") {
        eprintln!("dream64-server never compiles projects; use `dream64-compiler <world.dme>`");
        return ExitCode::from(2);
    } else if first.as_os_str() == OsStr::new("plan") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dream64-server plan <world.dme> [map.dmm]");
            return ExitCode::from(2);
        };
        (Command::Plan, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("boot") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dream64-server boot <world.dme> [map.dmm]");
            return ExitCode::from(2);
        };
        (Command::Boot, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("sweep") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dream64-server sweep <world.dme> [map.dmm]");
            return ExitCode::from(2);
        };
        (Command::Sweep, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("sweep-closure") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dream64-server sweep-closure <world.dme> [map.dmm]");
            return ExitCode::from(2);
        };
        (Command::SweepClosure, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("lobby-preflight") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dream64-server lobby-preflight <world.dme>");
            return ExitCode::from(2);
        };
        (Command::LobbyPreflight, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("lobby-preview") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dream64-server lobby-preview <world.dme> <world-params>");
            return ExitCode::from(2);
        };
        (Command::LobbyPreview, PathBuf::from(environment))
    } else {
        (Command::Plan, PathBuf::from(first))
    };
    let requested_map = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        eprintln!(
            "usage: dream64-server [plan|boot|sweep|sweep-closure|lobby-preflight] <world.dme> [map.dmm]"
        );
        return ExitCode::from(2);
    }
    if command == Command::Compile && requested_map.is_some() {
        eprintln!("usage: dm-lifecycle {} <world.dme>", "compile");
        return ExitCode::from(2);
    }
    let ready_world_mode = if command == Command::Boot {
        match ready_world_mode_from_environment() {
            Ok(mode) => mode,
            Err(error) => {
                eprintln!("ready-world mode: {error}");
                return ExitCode::from(2);
            }
        }
    } else {
        ReadyWorldMode::Disabled
    };
    let audit_runtime =
        command == Command::Boot && env::var_os("DREAM64_BOOT_AUDIT_RUNTIME").is_some();
    let mut boot_startup_ipc = if command == Command::Boot
        && !audit_runtime
        && !matches!(&ready_world_mode, ReadyWorldMode::Prewarm(_))
    {
        let ipc_address =
            env::var("DREAM64_IPC_ADDR").unwrap_or_else(|_| "0.0.0.0:51664".to_owned());
        let ipc = match parse_loopback_address(&ipc_address)
            .and_then(|address| LoopbackIpc::bind_starting(address, "Validating compiled project"))
        {
            Ok(ipc) => ipc,
            Err(error) => {
                eprintln!("loopback IPC: {error}");
                return ExitCode::FAILURE;
            }
        };
        eprintln!(
            "server-progress: loopback-ipc={} startup=validation",
            ipc.local_addr()
        );
        report_public_endpoint(ipc.local_addr().port());
        Some(ipc)
    } else {
        None
    };
    let compile_started = Instant::now();
    let cached_compilation = matches!(
        command,
        Command::Compile | Command::Boot | Command::LobbyPreflight | Command::LobbyPreview
    );
    if cached_compilation {
        let progress = progress_label(command);
        eprintln!(
            "{progress}: preparing compiled executable {}",
            environment.display()
        );
    }
    let standalone_artifact = environment
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("d64"));
    let cache_file = project_cache_file(&environment);
    let artifact_file = if standalone_artifact {
        environment.clone()
    } else {
        executable_artifact_file(&environment)
    };
    if standalone_artifact && command == Command::Boot {
        if requested_map.is_some() {
            eprintln!("standalone runtime artifacts use their linked map directory");
            return ExitCode::from(2);
        }
        return run_standalone_linked_boot(
            &artifact_file,
            audit_runtime,
            boot_startup_ipc.take(),
            process_started,
        );
    }
    let mut cached_executable = None;
    let mut cached_procedures = None;
    let mut cached_structural_seed = None;
    let mut cached_linked_runtime = None;
    let mut cached_lifecycle_index = None;
    let mut cached_default_map = None;
    let compilation_result: Result<_, String> = if cached_compilation {
        if standalone_artifact {
            prepare_standalone_artifact(&artifact_file)
        } else {
            prepare_compiled_executable(
                &environment,
                &cache_file,
                &artifact_file,
                command == Command::Compile,
            )
        }
        .map(|prepared| {
            let progress = progress_label(command);
            eprintln!(
                "{progress}: executable-artifact executable_artifact={} artifact={} miss_reason={:?} lowering_new={} lowering_deferred={} procedures={}",
                if prepared.artifact_hit { "hit" } else { "miss" },
                artifact_file.display(),
                prepared.miss_reason.as_deref().unwrap_or("none"),
                prepared.new_lowerings,
                prepared.executable.module().deferred_procedure_count(),
                prepared.executable.module().procedure_count(),
            );
            let cache = PreparedCacheStats {
                project_snapshot_hit: prepared.project_snapshot_hit,
                parsed_syntax_hit: prepared.parsed_syntax_hit,
                artifact_hit: prepared.artifact_hit,
            };
            cached_executable = Some(prepared.executable);
            cached_procedures = Some(prepared.procedures);
            cached_structural_seed = Some(prepared.structural_seed);
            cached_linked_runtime = prepared.linked_runtime;
            cached_lifecycle_index = prepared.lifecycle_index;
            cached_default_map = prepared.default_map;
            (prepared.compilation, Some(cache))
        })
    } else {
        CompilerDatabase::new()
            .compile(&environment)
            .map(|compilation| (compilation, None))
            .map_err(|error| error.to_string())
    };
    let (compilation, project_cache) = match compilation_result {
        Ok(compilation) => compilation,
        Err(error) => {
            eprintln!("{}: {error}", environment.display());
            return ExitCode::FAILURE;
        }
    };
    if cached_compilation {
        let progress = progress_label(command);
        eprintln!(
            "{progress}: runtime-image-ready elapsed_ms={} source_validation={} parsed_syntax_cache={} cache={}",
            compile_started.elapsed().as_millis(),
            if standalone_artifact {
                "not-required"
            } else if project_cache.is_some_and(|cache| cache.project_snapshot_hit) {
                "hit"
            } else {
                "miss"
            },
            project_cache
                .and_then(|cache| cache.parsed_syntax_hit)
                .map_or("skipped", |hit| if hit { "hit" } else { "miss" }),
            cache_file.display(),
        );
    }
    if let Some(selector) =
        env::var_os("DREAM64_DUMP_PROCEDURE").and_then(|value| value.into_string().ok())
        && let Some(executable) = cached_executable.as_ref()
    {
        for (index, path) in executable.module().procedure_paths().enumerate() {
            if !path.contains(&selector) {
                continue;
            }
            let Some(id) = executable.module().procedure_id_at(index) else {
                continue;
            };
            let Some(program) = executable.module().procedure(id) else {
                continue;
            };
            eprintln!(
                "procedure-dump: path={path} parameters={} locals={} instructions={}",
                program.parameter_count,
                program.local_count,
                program.instructions.len(),
            );
            for (pc, instruction) in program.instructions.iter().enumerate() {
                eprintln!(
                    "procedure-dump: pc={pc} source={:?} opcode={instruction:?}",
                    program.source_spans.get(pc),
                );
            }
        }
    }
    if command == Command::Compile {
        eprintln!(
            "compile-progress: artifact-ready project_files={} parsed_files={} definitions={} executable_artifact={} artifact={}",
            compilation.stats().project_files,
            compilation.stats().parsed_files,
            compilation.stats().definitions,
            if project_cache.is_some_and(|cache| cache.artifact_hit) {
                "hit"
            } else {
                "miss"
            },
            artifact_file.display(),
        );
        return ExitCode::SUCCESS;
    }
    let procedures = cached_procedures
        .take()
        .unwrap_or_else(|| ProcedureRegistry::build(&compilation));
    let mut prepared_boot = None;
    if command == Command::Boot {
        if let Some(ipc) = &boot_startup_ipc {
            ipc.commit_boot_phase(
                ipc.startup_generation(),
                BootPhase::StructuralLoad,
                "Loading and parsing the map",
            )
            .expect("boot phase advances monotonically");
        }
        eprintln!("boot-progress: loading map");
        let linked_map = requested_map
            .is_none()
            .then(|| cached_default_map.take())
            .flatten();
        let (map_path, map_source) = match linked_map {
            Some((map_path, world)) => {
                let ready_cache = cache_file.with_extension("linked-map.ready");
                let compile_index = cached_lifecycle_index.clone().unwrap_or_else(|| {
                    LifecycleIndex::build_compile_only(&compilation, &procedures)
                });
                let precompiled = match cached_executable.take() {
                    Some(executable) => precompile_lifecycle_for_world_with_executable(
                        &compilation,
                        &procedures,
                        &compile_index,
                        &world,
                        executable,
                    ),
                    None => match precompile_lifecycle_for_world(
                        &compilation,
                        &procedures,
                        &compile_index,
                        &world,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            eprintln!("lifecycle precompile: {error}");
                            return ExitCode::FAILURE;
                        }
                    },
                };
                prepared_boot = Some((map_path, world, precompiled, ready_cache));
                (String::new(), String::new())
            }
            None => match load_map(&compilation, requested_map.as_deref()) {
                Ok(map) => map,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::FAILURE;
                }
            },
        };
        if prepared_boot.is_some() {
            // The linked map already supplied the complete world/precompile tuple.
        } else {
            eprintln!("boot-progress: preparing map plan {map_path}");
            let ready_cache = ready_world_cache_file(
                &cache_file,
                &map_source,
                &compilation,
                ready_world_mode.production_identity(),
            );
            let world = match cached_world_plan(&cache_file, &map_source, &compilation) {
                Ok(world) => world,
                Err(error) => {
                    eprintln!("{map_path}: {error}");
                    return ExitCode::FAILURE;
                }
            };
            drop(map_source);
            let compile_index = cached_lifecycle_index
                .clone()
                .unwrap_or_else(|| LifecycleIndex::build_compile_only(&compilation, &procedures));
            if let Some(ipc) = &boot_startup_ipc {
                ipc.commit_boot_phase(
                    ipc.startup_generation(),
                    BootPhase::StructuralLoad,
                    "Precompiling lifecycle procedures",
                )
                .expect("boot phase advances monotonically");
            }
            eprintln!("boot-progress: precompiling lifecycle before runtime materialization");
            let started = Instant::now();
            let precompiled = match cached_executable.take() {
                Some(executable) => Ok(precompile_lifecycle_for_world_with_executable(
                    &compilation,
                    &procedures,
                    &compile_index,
                    &world,
                    executable,
                )),
                None => precompile_lifecycle_for_world(
                    &compilation,
                    &procedures,
                    &compile_index,
                    &world,
                ),
            };
            let precompiled = match precompiled {
                Ok(precompiled) => precompiled,
                Err(error) => {
                    eprintln!("lifecycle precompile: {error}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!(
                "boot-progress: lifecycle-precompile-complete elapsed_ms={} targets={} bodies={} procedures={} deferred={}",
                started.elapsed().as_millis(),
                precompiled.targets(),
                precompiled.reachable_bodies(),
                precompiled.module_procedures(),
                precompiled.deferred_procedures(),
            );
            prepared_boot = Some((map_path, world, precompiled, ready_cache));
        }
    }
    if command == Command::Boot {
        if let Some(ipc) = &boot_startup_ipc {
            ipc.commit_boot_phase(
                ipc.startup_generation(),
                BootPhase::StructuralLoad,
                "Materializing globals and type defaults",
            )
            .expect("boot phase advances monotonically");
        }
        eprintln!("boot-progress: materializing globals and type defaults");
    }
    let runtime_observer = |event: RuntimeImageConstructionEvent| {
        if command != Command::Boot {
            return;
        }
        if event.completed {
            eprintln!(
                "boot-progress: runtime-phase-complete phase={} elapsed_ms={} items={}",
                event.phase.as_str(),
                event.elapsed.as_millis(),
                event
                    .items
                    .map_or_else(|| "unknown".to_owned(), |items| items.to_string()),
            );
        } else {
            eprintln!(
                "boot-progress: runtime-phase-start phase={}",
                event.phase.as_str()
            );
        }
    };
    let runtime_result = if let Some(runtime) = cached_linked_runtime.take() {
        Ok(runtime)
    } else if let Some((_, _, precompiled, _)) = prepared_boot.as_mut() {
        match cached_structural_seed.take() {
            Some(seed) => RuntimeImage::from_compilation_with_prelinked_module_and_seed(
                &compilation,
                &procedures,
                precompiled.module_mut_for_runtime_initializers(),
                seed,
                runtime_observer,
            ),
            None => RuntimeImage::from_compilation_with_prelinked_module(
                &compilation,
                &procedures,
                precompiled.module_mut_for_runtime_initializers(),
                runtime_observer,
            ),
        }
    } else {
        match cached_structural_seed.take() {
            Some(seed) => {
                RuntimeImage::from_compilation_with_seed(&compilation, seed, runtime_observer)
            }
            None => RuntimeImage::from_compilation_with_observer(&compilation, runtime_observer),
        }
    };
    let mut runtime = match runtime_result {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("runtime image: {error}");
            return ExitCode::FAILURE;
        }
    };
    if command == Command::Boot {
        if let ReadyWorldMode::Activate(identity) = &ready_world_mode {
            eprintln!(
                "boot-progress: random stream preserved from ready-world snapshot deployment={:?} seed={}",
                identity.deployment_id, identity.random_seed,
            );
        } else {
            let (launch_random_seed, random_seed_source) = match &ready_world_mode {
                ReadyWorldMode::Prewarm(identity) => (identity.random_seed, "production-identity"),
                _ => launch_random_seed(),
            };
            runtime.set_launch_random_seed(launch_random_seed);
            eprintln!(
                "boot-progress: random stream seeded for this launch source={random_seed_source} seed={launch_random_seed}"
            );
        }
        let stats = runtime.stats();
        eprintln!(
            "boot-progress: initializer frontier selectors={} typed_constructors={} dynamic_constructor_fallback={} complete_inventory_fallback={} module_procedures={} deferred={} materialized={} direct_initial_values={} shared_reflection_entries={}",
            stats.initializer_frontier_selectors,
            stats.initializer_typed_constructor_targets,
            stats.initializer_dynamic_constructor_frontier,
            stats.initializer_complete_symbol_inventory,
            stats.initializer_module_procedures,
            stats.initializer_module_deferred_procedures,
            stats.initializer_module_materialized_procedures,
            stats.execution_initial_value_entries,
            stats.shared_reflection_entries,
        );
    }
    for diagnostic in runtime.diagnostics() {
        eprintln!("{}", format_runtime_diagnostic(diagnostic));
    }
    if matches!(command, Command::LobbyPreflight | Command::LobbyPreview) {
        let Some(executable) = cached_executable.take() else {
            eprintln!("lobby-preflight: compiled executable is unavailable");
            return ExitCode::FAILURE;
        };
        let Some(world_params) = requested_map.as_deref().and_then(Path::to_str) else {
            eprintln!(
                "usage: dm-lifecycle lobby-preflight <world.dme> <world-params>\n\
                 supply the project's supported no-startup parameter so world/New returns before subsystem boot"
            );
            return ExitCode::from(2);
        };
        let index = cached_lifecycle_index
            .clone()
            .unwrap_or_else(|| LifecycleIndex::build(&compilation, &procedures, &runtime));
        return run_lobby_preflight(
            &environment,
            world_params,
            &index,
            &mut runtime,
            &executable,
            command == Command::LobbyPreview,
        );
    }
    if command == Command::Boot {
        if let Some(ipc) = &boot_startup_ipc {
            ipc.commit_boot_phase(
                ipc.startup_generation(),
                BootPhase::StructuralLoad,
                "Indexing lifecycle dispatch",
            )
            .expect("boot phase advances monotonically");
        }
        eprintln!("boot-progress: indexing procedures and lifecycle dispatch");
    }
    let index = cached_lifecycle_index
        .take()
        .unwrap_or_else(|| LifecycleIndex::build(&compilation, &procedures, &runtime));
    let mut boot_precompiled = None;
    let mut ready_world_cache = None;
    let (map_path, world) =
        if let Some((map_path, world, precompiled, ready_cache)) = prepared_boot.take() {
            boot_precompiled = Some(precompiled);
            ready_world_cache = Some(ready_cache);
            (map_path, world)
        } else {
            let (map_path, map_source) = match load_map(&compilation, requested_map.as_deref()) {
                Ok(map) => map,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::FAILURE;
                }
            };
            let world = match cached_world_plan(&cache_file, &map_source, &compilation) {
                Ok(world) => world,
                Err(error) => {
                    eprintln!("{map_path}: {error}");
                    return ExitCode::FAILURE;
                }
            };
            (map_path, world)
        };
    let plan = build_initialization_plan(&runtime, &index, &world, map_path.clone());

    print_plan_summary(&map_path, &index, &procedures, &plan);
    if command == Command::Sweep {
        print_compatibility_sweep(&sweep_lifecycle_compatibility(
            &compilation,
            &procedures,
            &index,
            &plan,
        ));
    }
    if command == Command::SweepClosure {
        print_compatibility_sweep(&sweep_lifecycle_compatibility_with_closures(
            &compilation,
            &procedures,
            &index,
            &plan,
        ));
    }
    if command == Command::Boot {
        let readiness = master_controller_readiness(&compilation, &runtime);
        let lobby_readiness = lobby_pregame_readiness(&compilation, &runtime);
        if let Some(probe) = &readiness {
            eprintln!(
                "boot-progress: readiness probe global={} fields={} expected={:?}",
                probe.global,
                probe.fields.len(),
                probe.expected
            );
        }
        // The executable lifecycle module, RuntimeImage, LifecycleIndex, and
        // initialization plan own all data required from this point onward.
        // Retaining the full syntax compilation and procedure registry across
        // dynamic station-map loading needlessly keeps the cold compiler graph
        // resident at the server's peak heap size.
        drop(procedures);
        drop(compilation);
        let mut startup_ipc = boot_startup_ipc.take();
        let ready_cache = ready_world_cache
            .as_deref()
            .expect("production boot prepared a ready-world cache identity");
        let restore_snapshot = matches!(
            &ready_world_mode,
            ReadyWorldMode::Development | ReadyWorldMode::Activate(_)
        );
        if restore_snapshot && ready_cache.is_file() {
            if let Some(ipc) = &startup_ipc {
                ipc.commit_boot_phase(
                    ipc.startup_generation(),
                    BootPhase::WorldPlan,
                    "Restoring ready world",
                )
                .expect("boot phase advances monotonically");
            }
            let restore_started = Instant::now();
            let mut state = runtime.take_execution_state();
            let precompiled = boot_precompiled.as_mut().expect("boot precompile exists");
            match restore_ready_world_cache(ready_cache, &mut state, precompiled.module()) {
                Ok(bytes) => {
                    if matches!(&ready_world_mode, ReadyWorldMode::Development) {
                        // Development reuse intentionally starts a fresh random
                        // stream. Production activation instead continues the
                        // exact stream captured by its matching prewarm image.
                        state.reseed_random(fresh_launch_random_seed());
                    }
                    precompiled.install_persistent_state(state);
                    eprintln!(
                        "boot-progress: ready-world-cache cache=hit artifact={} bytes={} restore_ms={}",
                        ready_cache.display(),
                        bytes,
                        restore_started.elapsed().as_millis(),
                    );
                    eprintln!(
                        "boot-progress: headless ready from snapshot; entering persistent scheduler loop ready_elapsed_ms={}",
                        process_started.elapsed().as_millis()
                    );
                    return run_persistent_server_loop(
                        &mut runtime,
                        precompiled,
                        startup_ipc,
                        lobby_readiness.as_ref(),
                    );
                }
                Err(error) => {
                    runtime.restore_execution_state(state);
                    if matches!(&ready_world_mode, ReadyWorldMode::Activate(_)) {
                        eprintln!(
                            "boot-progress: ready-world activation failed closed artifact={} reason={error:?}",
                            ready_cache.display(),
                        );
                        return ExitCode::FAILURE;
                    }
                    eprintln!(
                        "boot-progress: ready-world-cache cache=miss artifact={} reason={error:?}",
                        ready_cache.display(),
                    );
                    if let Err(remove_error) = fs::remove_file(ready_cache) {
                        eprintln!(
                            "boot-progress: ready-world-cache corrupt-artifact-retained error={remove_error}"
                        );
                    }
                }
            }
        } else {
            if matches!(&ready_world_mode, ReadyWorldMode::Activate(_)) {
                eprintln!(
                    "boot-progress: ready-world activation failed closed artifact={} reason=not-found",
                    ready_cache.display(),
                );
                return ExitCode::FAILURE;
            }
            eprintln!(
                "boot-progress: ready-world-cache cache=miss artifact={} reason={:?}",
                ready_cache.display(),
                match &ready_world_mode {
                    ReadyWorldMode::Disabled => "disabled",
                    ReadyWorldMode::Prewarm(_) => "production prewarm requested",
                    ReadyWorldMode::Development | ReadyWorldMode::Activate(_) => "not found",
                },
            );
        }
        if let Some(ipc) = &startup_ipc {
            ipc.commit_boot_phase(
                ipc.startup_generation(),
                BootPhase::WorldPlan,
                "Preflighting map initializer plans",
            )
            .expect("boot phase advances monotonically");
        }
        eprintln!("boot-progress: preflighting map initializer plans");
        let map_types = world
            .templates()
            .values()
            .flat_map(|template| template.initializers.iter())
            .filter_map(|initializer| {
                matches!(
                    initializer.resolution,
                    dm_world::InitializerResolution::Resolved { .. }
                )
                .then(|| dm_value::TypePath::parse(&initializer.path).ok())
                .flatten()
            })
            .collect::<Vec<_>>();
        match runtime.preflight_instance_initializers(map_types) {
            Ok(stats) => eprintln!(
                "boot-progress: initializer preflight complete types={} compiled={} reused={}",
                stats.types, stats.plans_compiled, stats.plans_reused
            ),
            Err(errors) => {
                eprintln!(
                    "initializer preflight failed: {} type plan error(s)",
                    errors.len()
                );
                for error in errors {
                    eprintln!("initializer preflight: {error}");
                }
                return ExitCode::FAILURE;
            }
        }
        if let Some(ipc) = &startup_ipc {
            ipc.commit_boot_phase(
                ipc.startup_generation(),
                BootPhase::WorldAllocation,
                "Allocating the map world",
            )
            .expect("boot phase advances monotonically");
        }
        eprintln!("boot-progress: allocating map world");
        let allocation = match allocate_world(&world, &mut runtime) {
            Ok(allocation) => allocation,
            Err(error) => {
                eprintln!("world allocation: {error}");
                return ExitCode::FAILURE;
            }
        };
        // ExecutionState already retains the shared compiled initializer
        // catalog used by VM `new`. These RuntimeImage-only allocation caches
        // serve host-side bulk allocation and can be dropped without causing
        // Genesis constructors to rebuild them.
        let released = runtime.release_allocation_caches();
        eprintln!(
            "boot-progress: released allocation caches initializer_plans={} initializer_programs={} datum_plans={}",
            released.initializer_plans, released.initializer_programs, released.allocation_plans,
        );
        // Allocation is the last consumer of the coordinate/template plan.
        // Runtime execution uses the compact allocation and initialization
        // event plan, so release the map plan before compiling lifecycle code.
        drop(world);
        let precompiled = boot_precompiled.as_mut().expect("boot preparation exists");
        if let Some(ipc) = &startup_ipc {
            ipc.commit_boot_phase(
                ipc.startup_generation(),
                BootPhase::Lifecycle,
                "Starting world and subsystem controller",
            )
            .expect("boot phase advances monotonically");
        }
        let startup_limits = startup_scheduler_limits();
        let execution = match if audit_runtime {
            audit_initialization_plan_with_precompiled(
                &index,
                &plan,
                &allocation,
                &mut runtime,
                startup_limits,
                precompiled,
            )
        } else if let Some(ipc) = startup_ipc.as_mut() {
            let mut service_startup_clients =
                |_executable: &ExecutableProcedures, _state: &mut dm_vm::ExecutionState| {
                    // Keep the listener available for phase polling, but do
                    // not materialize /client datums in the world that will
                    // become the content-addressed ready image. The native
                    // client retries its attach once per second and replaces
                    // its startup replay as soon as the gate opens below.
                    ipc.commit_boot_phase(
                        ipc.startup_generation(),
                        BootPhase::Lifecycle,
                        "Initializing subsystems",
                    )
                    .expect("boot phase advances monotonically");
                };
            execute_boot_initialization_plan_with_precompiled_and_startup_service(
                &index,
                &plan,
                &allocation,
                &mut runtime,
                startup_limits,
                readiness.as_ref(),
                precompiled,
                &mut service_startup_clients,
            )
        } else {
            execute_boot_initialization_plan_with_precompiled(
                &index,
                &plan,
                &allocation,
                &mut runtime,
                startup_limits,
                readiness.as_ref(),
                precompiled,
            )
        } {
            Ok(execution) => execution,
            Err(error) => {
                eprintln!("initialization: {error}");
                return ExitCode::FAILURE;
            }
        };
        print_boot_summary(&allocation, &execution);
        if execution.scheduler.termination != SchedulerDrainTermination::HeadlessReady {
            eprintln!(
                "initialization stopped before authoritative readiness: {:?}",
                execution.scheduler.termination
            );
            return ExitCode::FAILURE;
        }
        if let Some(state) = precompiled.persistent_state_mut() {
            inspect_ready_globals(state);
            let compaction = state.compact_quiescent_heap();
            eprintln!(
                "boot-progress: ready-heap-compaction reclaimed_datums={} reclaimed_lists={} elapsed_ms={}",
                compaction.reclaimed_datums,
                compaction.reclaimed_lists,
                compaction.elapsed.as_millis(),
            );
        }
        eprintln!(
            "boot-progress: headless ready; entering persistent scheduler loop ready_elapsed_ms={}",
            process_started.elapsed().as_millis()
        );
        let mut snapshot_written = false;
        if ready_world_mode.writes_snapshot()
            && let Some(state) = precompiled.persistent_state_mut()
        {
            let snapshot_started = Instant::now();
            match write_ready_world_cache(ready_cache, state) {
                Ok(bytes) => {
                    snapshot_written = true;
                    eprintln!(
                        "boot-progress: ready-world-cache cache=stored artifact={} bytes={} write_ms={}",
                        ready_cache.display(),
                        bytes,
                        snapshot_started.elapsed().as_millis(),
                    );
                }
                Err(error) => {
                    eprintln!(
                        "boot-progress: ready-world-cache store-failed artifact={} error={error:?}",
                        ready_cache.display(),
                    );
                    if matches!(&ready_world_mode, ReadyWorldMode::Prewarm(_)) {
                        return ExitCode::FAILURE;
                    }
                }
            }
        }
        if let ReadyWorldMode::Prewarm(identity) = &ready_world_mode {
            if !snapshot_written {
                eprintln!(
                    "boot-progress: ready-world prewarm failed artifact={} reason=persistent-state-unavailable",
                    ready_cache.display(),
                );
                return ExitCode::FAILURE;
            }
            eprintln!(
                "boot-progress: ready-world prewarm complete artifact={} deployment={:?} seed={} elapsed_ms={}",
                ready_cache.display(),
                identity.deployment_id,
                identity.random_seed,
                process_started.elapsed().as_millis(),
            );
            if let Some(control_address) = env::var_os("DREAM64_PREWARM_STANDBY_ADDR") {
                let Some(control_address) = control_address.to_str() else {
                    eprintln!("DREAM64_PREWARM_STANDBY_ADDR is not valid Unicode");
                    return ExitCode::FAILURE;
                };
                return run_prewarmed_standby(
                    &mut runtime,
                    precompiled,
                    identity,
                    control_address,
                    lobby_readiness.as_ref(),
                );
            }
            return ExitCode::SUCCESS;
        }
        return run_persistent_server_loop(
            &mut runtime,
            precompiled,
            startup_ipc,
            lobby_readiness.as_ref(),
        );
    }
    ExitCode::SUCCESS
}

/// Prints selected live global datum fields at the authoritative-ready
/// boundary. This is deliberately opt-in: production servers retain the same
/// output and behavior, while compatibility investigations can inspect the
/// actual VM heap without adding game-specific knowledge to the engine.
///
/// Syntax: `DREAM64_READY_INSPECT=SSmapping:z_list,z_level_to_stack;Master:processing`.
fn inspect_ready_globals(state: &dm_vm::ExecutionState) {
    let Some(specification) = env::var_os("DREAM64_READY_INSPECT")
        .and_then(|value| value.into_string().ok())
        .filter(|value| !value.trim().is_empty())
    else {
        return;
    };
    for entry in specification
        .split(';')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (global_name, fields) = entry.split_once(':').unwrap_or((entry, ""));
        let Ok(global) = FieldName::parse(global_name.trim()) else {
            eprintln!("ready-inspect: global={global_name} error=invalid-global-name");
            continue;
        };
        let value = state.global(&global).cloned().unwrap_or(Value::Null);
        eprintln!(
            "ready-inspect: global={} value={}",
            global,
            inspected_value(state, &value),
        );
        let Value::Datum(datum) = value else {
            continue;
        };
        for field_name in fields
            .split(',')
            .map(str::trim)
            .filter(|field| !field.is_empty())
        {
            let Ok(field) = FieldName::parse(field_name) else {
                eprintln!(
                    "ready-inspect: global={} field={} error=invalid-field-name",
                    global, field_name,
                );
                continue;
            };
            match state.heap().datum_field(datum, &field) {
                Ok(value) => eprintln!(
                    "ready-inspect: global={} field={} value={}",
                    global,
                    field,
                    inspected_value(state, value),
                ),
                Err(error) => eprintln!(
                    "ready-inspect: global={} field={} error={error}",
                    global, field,
                ),
            }
        }
    }
}

fn inspected_value(state: &dm_vm::ExecutionState, value: &Value) -> String {
    match value {
        Value::List(list) => state.heap().list(*list).map_or_else(
            |error| format!("list({list:?}, error={error})"),
            |value| format!("list({list:?}, len={})", value.len()),
        ),
        Value::Datum(datum) => state.heap().datum(*datum).map_or_else(
            |error| format!("datum({datum:?}, error={error})"),
            |value| format!("datum({datum:?}, type={})", value.type_path()),
        ),
        value => format!("{value:?}"),
    }
}

fn run_prewarmed_standby(
    runtime: &mut RuntimeImage,
    precompiled: &mut dm_lifecycle::PrecompiledLifecycle,
    identity: &ProductionReadyWorldIdentity,
    control_address: &str,
    lobby_readiness: Option<&HeadlessReadinessProbe>,
) -> ExitCode {
    let control_address = match parse_loopback_address(control_address) {
        Ok(address) => address,
        Err(error) => {
            eprintln!("prewarm standby address: {error}");
            return ExitCode::FAILURE;
        }
    };
    let listener = match TcpListener::bind(control_address) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("prewarm standby bind {control_address}: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "boot-progress: prewarmed standby ready control={} deployment={:?} seed={}",
        control_address, identity.deployment_id, identity.random_seed,
    );
    let expected = format!("ACTIVATE {}", identity.deployment_id);
    loop {
        let (stream, peer) = match listener.accept() {
            Ok(connection) => connection,
            Err(error) => {
                eprintln!("prewarm standby accept: {error}");
                return ExitCode::FAILURE;
            }
        };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
        let mut command = String::new();
        match BufReader::new(stream).take(4_096).read_line(&mut command) {
            Ok(_) if command.trim() == expected => {
                eprintln!(
                    "boot-progress: prewarmed standby activation accepted peer={peer} deployment={:?}",
                    identity.deployment_id,
                );
                break;
            }
            Ok(_) if command.trim() == format!("CANCEL {}", identity.deployment_id) => {
                eprintln!(
                    "boot-progress: prewarmed standby cancelled deployment={:?}",
                    identity.deployment_id,
                );
                return ExitCode::SUCCESS;
            }
            Ok(_) => eprintln!("prewarm standby rejected command from {peer}"),
            Err(error) => eprintln!("prewarm standby read from {peer}: {error}"),
        }
    }
    drop(listener);

    let ipc_address = env::var("DREAM64_IPC_ADDR").unwrap_or_else(|_| "0.0.0.0:51664".to_owned());
    let ipc_address = match parse_loopback_address(&ipc_address) {
        Ok(address) => address,
        Err(error) => {
            eprintln!("loopback IPC: {error}");
            return ExitCode::FAILURE;
        }
    };
    let timeout = env::var("DREAM64_HANDOFF_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map_or(Duration::from_secs(30), Duration::from_millis);
    let deadline = Instant::now() + timeout;
    let ipc = loop {
        match LoopbackIpc::bind_starting(ipc_address, "Activating prepared world") {
            Ok(ipc) => break ipc,
            Err(error) if Instant::now() < deadline => {
                eprintln!(
                    "boot-progress: handoff waiting for ipc={} reason={error}",
                    ipc_address
                );
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                eprintln!(
                    "boot-progress: handoff failed ipc={} reason={error}",
                    ipc_address
                );
                return ExitCode::FAILURE;
            }
        }
    };
    report_public_endpoint(ipc.local_addr().port());
    eprintln!(
        "boot-progress: prewarmed handoff complete ipc={} deployment={:?}",
        ipc.local_addr(),
        identity.deployment_id,
    );
    run_persistent_server_loop(runtime, precompiled, Some(ipc), lobby_readiness)
}

fn launch_random_seed() -> (u64, &'static str) {
    launch_random_seed_from(env::var("DREAM64_RANDOM_SEED").ok().as_deref())
}

fn launch_random_seed_from(value: Option<&str>) -> (u64, &'static str) {
    if let Some(value) = value
        && let Ok(seed) = value.parse::<u64>()
        && seed != 0
    {
        return (seed, "environment");
    }
    (fresh_launch_random_seed(), "host-entropy")
}

fn fresh_launch_random_seed() -> u64 {
    // `RandomState::new()` obtains independently keyed entropy from the host.
    // Mix in launch-local values as domain separation and avoid the all-zero
    // state used by deterministic unit tests.
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u128(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
    );
    hasher.write_u32(std::process::id());
    let seed = hasher.finish();
    if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    }
}

fn run_persistent_server_loop(
    runtime: &mut RuntimeImage,
    precompiled: &mut dm_lifecycle::PrecompiledLifecycle,
    startup_ipc: Option<LoopbackIpc>,
    lobby_readiness: Option<&HeadlessReadinessProbe>,
) -> ExitCode {
    let mut ipc_address = startup_ipc.expect("production boot bound loopback IPC");
    if let Err(error) =
        activate_lobby_generation(runtime, precompiled, &mut ipc_address, lobby_readiness)
    {
        eprintln!("generation activation: {error}");
        return ExitCode::FAILURE;
    }
    ipc_address
        .commit_boot_phase(
            ipc_address.startup_generation(),
            BootPhase::RuntimeStartedReadOnly,
            "Lobby ready — round countdown active",
        )
        .expect("activated lobby belongs to current boot generation");
    let startup_generation = ipc_address.startup_generation();
    if let Err(error) = ipc_address.accept_startup_clients(startup_generation) {
        eprintln!("generation activation commit: {error}");
        return ExitCode::FAILURE;
    }
    if let Some(state) = precompiled.persistent_state_mut() {
        ipc_address.enable_session_interaction(state);
    }
    eprintln!(
        "server-progress: loopback-ipc={} startup=accepting lobby=pregame",
        ipc_address.local_addr()
    );
    let max_slices = env::var_os("DREAM64_BOOT_MAX_SLICES")
        .and_then(|limit| {
            limit
                .to_str()
                .and_then(|value| value.parse::<u64>().ok())
                .map(Some)
                .or_else(|| {
                    eprintln!("DREAM64_BOOT_MAX_SLICES ignored: not a valid u64: {limit:?}");
                    None
                })
        })
        .flatten();
    let mut slices = 0u64;
    let mut host_budget = HostSliceBudget::new(100_000, 1_000, 100_000, Duration::from_millis(10));
    let mut max_vm_slice = Duration::ZERO;
    let mut over_target_slices = 0u64;
    loop {
        let slice_started = Instant::now();
        let tick_duration = precompiled.persistent_tick_duration();
        ipc_address.apply_lifecycle_tick_boundary(precompiled);
        let vm_started = Instant::now();
        let scheduled_steps = host_budget.steps();
        let scheduler = match advance_persistent_scheduler_responsive(
            precompiled,
            runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 1,
            },
            scheduled_steps,
        ) {
            Ok(scheduler) => scheduler,
            Err(error) => {
                eprintln!("persistent scheduler: {error}");
                return ExitCode::FAILURE;
            }
        };
        let vm_elapsed = vm_started.elapsed();
        max_vm_slice = max_vm_slice.max(vm_elapsed);
        over_target_slices =
            over_target_slices.saturating_add(u64::from(vm_elapsed > Duration::from_millis(10)));
        host_budget.observe(vm_elapsed);
        ipc_address.apply_lifecycle_tick_boundary(precompiled);
        slices = slices.saturating_add(1);
        if slices == 1 || slices % 100 == 0 || vm_elapsed >= Duration::from_millis(50) {
            eprintln!(
                "server-progress: scheduler slice={} tick={} rounds={} completed={} failed={} pending={} termination={:?} vm_us={} vm_max_us={} next_step_budget={} over_10ms={} host_loop_us={}",
                slices,
                scheduler.final_tick,
                scheduler.rounds,
                scheduler.completed_tasks,
                scheduler.failed_tasks,
                scheduler.pending_tasks,
                scheduler.termination,
                vm_elapsed.as_micros(),
                max_vm_slice.as_micros(),
                host_budget.steps(),
                over_target_slices,
                slice_started.elapsed().as_micros(),
            );
        }
        if let Some(limit) = max_slices
            && slices >= limit
        {
            eprintln!("boot-progress: reached DREAM64_BOOT_MAX_SLICES={limit}; stopping");
            for line in precompiled.bounded_scheduler_progress() {
                eprintln!("boot-progress: shutdown-dm-frame {line}");
            }
            return ExitCode::SUCCESS;
        }
        if let Some(remaining) = tick_duration.checked_sub(slice_started.elapsed()) {
            std::thread::sleep(remaining);
        }
    }
}

fn activate_lobby_generation(
    runtime: &mut RuntimeImage,
    precompiled: &mut dm_lifecycle::PrecompiledLifecycle,
    ipc: &mut LoopbackIpc,
    lobby_readiness: Option<&HeadlessReadinessProbe>,
) -> Result<(), String> {
    let Some(lobby_readiness) = lobby_readiness else {
        return Ok(());
    };
    let is_ready = |precompiled: &mut dm_lifecycle::PrecompiledLifecycle| {
        precompiled
            .persistent_state_mut()
            .is_some_and(|state| readiness_probe_matches(state, lobby_readiness))
    };
    if is_ready(precompiled) {
        return Ok(());
    }

    ipc.commit_boot_phase(
        ipc.startup_generation(),
        BootPhase::Lifecycle,
        "Finishing subsystem initialization",
    )?;
    ipc.show_startup_lobby(ipc.startup_generation())?;
    eprintln!(
        "boot-progress: generation activation gate opened read_only=true target=SSticker.current_state"
    );

    // A reconnecting native client retries once per second. Give it a short,
    // bounded window to enter the restored lobby before Monk's final
    // Master.Initialize tail performs population-sensitive storyteller setup.
    let attach_grace = env::var("DREAM64_ACTIVATION_ATTACH_GRACE_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(3_000);
    let grace_deadline = Instant::now() + Duration::from_millis(attach_grace);
    while Instant::now() < grace_deadline {
        ipc.apply_lifecycle_tick_boundary(precompiled);
        std::thread::sleep(Duration::from_millis(10));
    }

    let max_slices = env::var("DREAM64_ACTIVATION_MAX_SLICES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1_000);
    let target_millis = env::var("DREAM64_SCHEDULER_WALL_BUDGET_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(50)
        .max(1);
    // Native quickening must translate into more startup progress, not merely
    // finish the same fixed 100k legacy instructions earlier. The VM's wall
    // deadline remains authoritative for responsiveness; this controller
    // raises the logical ceiling while slices are comfortably under it.
    let mut activation_budget = HostSliceBudget::new(
        100_000,
        10_000,
        10_000_000,
        Duration::from_millis(target_millis),
    );
    for slice in 1..=max_slices {
        ipc.apply_lifecycle_tick_boundary(precompiled);
        if is_ready(precompiled) {
            eprintln!(
                "boot-progress: generation activation complete slices={} lobby=pregame",
                slice - 1,
            );
            return Ok(());
        }
        let vm_started = Instant::now();
        let scheduled_steps = activation_budget.steps();
        let scheduler = advance_persistent_scheduler_responsive(
            precompiled,
            runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 1,
            },
            scheduled_steps,
        )
        .map_err(|error| error.to_string())?;
        let vm_elapsed = vm_started.elapsed();
        activation_budget.observe(vm_elapsed);
        if slice == 1 || slice % 100 == 0 {
            eprintln!(
                "boot-progress: generation activation slice={slice} tick={} rounds={} completed={} failed={} pending={} termination={:?} vm_us={} step_budget={} next_step_budget={}",
                scheduler.final_tick,
                scheduler.rounds,
                scheduler.completed_tasks,
                scheduler.failed_tasks,
                scheduler.pending_tasks,
                scheduler.termination,
                vm_elapsed.as_micros(),
                scheduled_steps,
                activation_budget.steps(),
            );
        }
        ipc.apply_lifecycle_tick_boundary(precompiled);
        if is_ready(precompiled) {
            eprintln!(
                "boot-progress: generation activation complete slices={slice} tick={} completed={} pending={} lobby=pregame",
                scheduler.final_tick, scheduler.completed_tasks, scheduler.pending_tasks,
            );
            return Ok(());
        }
    }
    for line in precompiled.bounded_scheduler_progress() {
        eprintln!("boot-progress: activation-timeout-dm-frame {line}");
    }
    let (ruin_batches, ruin_steps) = dm_vm::native_ruin_batch_metrics();
    let (ruin_scan_activations, ruin_scan_cells, ruin_scan_rejections, ruin_scan_successes) =
        dm_vm::native_ruin_scan_metrics();
    let (ruin_flag_rejections, ruin_area_rejections) = dm_vm::native_ruin_rejection_causes();
    let (tgm_cells, tgm_safepoints, tgm_commits) = dm_vm::native_tgm_load_metrics();
    let (tgm_target_resolutions, tgm_target_cache_hits) = dm_vm::native_tgm_target_cache_metrics();
    let (tgm_build_cache_members, tgm_build_cache_logical_steps) =
        dm_vm::native_tgm_build_cache_metrics();
    let (build_coordinate_prefixes, build_coordinate_fallbacks) =
        dm_vm::native_build_coordinate_prefix_metrics();
    eprintln!(
        "boot-progress: native-quickening tgm_load_activations={} tgm_cells={} tgm_safepoints={} tgm_commits={} tgm_target_resolutions={} tgm_target_cache_hits={} tgm_build_cache_members={} tgm_build_cache_logical_steps={} build_coordinate_prefixes={} build_coordinate_fallbacks={} discover_offset_activations={} ruin_batches={} ruin_logical_steps={} ruin_scan_activations={} ruin_scan_cells={} ruin_scan_rejections={} ruin_scan_successes={} ruin_flag_rejections={} ruin_area_rejections={} ruin_rejection_cache_hits={}",
        dm_vm::native_tgm_load_activations(),
        tgm_cells,
        tgm_safepoints,
        tgm_commits,
        tgm_target_resolutions,
        tgm_target_cache_hits,
        tgm_build_cache_members,
        tgm_build_cache_logical_steps,
        build_coordinate_prefixes,
        build_coordinate_fallbacks,
        dm_vm::native_discover_offset_activations(),
        ruin_batches,
        ruin_steps,
        ruin_scan_activations,
        ruin_scan_cells,
        ruin_scan_rejections,
        ruin_scan_successes,
        ruin_flag_rejections,
        ruin_area_rejections,
        dm_vm::native_ruin_rejection_cache_hits(),
    );
    for sample in dm_vm::native_tgm_commit_samples() {
        eprintln!("boot-progress: tgm-commit {sample}");
    }
    for sample in dm_vm::native_tgm_continuation_rejections() {
        eprintln!("boot-progress: tgm-continuation-rejection {sample}");
    }
    for sample in dm_vm::native_tgm_route_samples() {
        eprintln!("boot-progress: tgm-route {sample}");
    }
    for sample in dm_vm::native_ruin_area_rejection_samples() {
        eprintln!("boot-progress: ruin-area-rejection {sample}");
    }
    Err(format!(
        "Monke lobby did not enter pregame within {max_slices} activation slices"
    ))
}

fn restore_ready_world_cache(
    path: &Path,
    state: &mut dm_vm::ExecutionState,
    module: &dm_vm::Module,
) -> Result<u64, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let bytes = file.metadata().map_err(|error| error.to_string())?.len();
    state
        .restore_ready_world_snapshot_from(&mut GzDecoder::new(BufReader::new(file)), module)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn write_ready_world_cache(path: &Path, state: &dm_vm::ExecutionState) -> Result<u64, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut temporary = None;
    for sequence in 0..100_u32 {
        let candidate =
            path.with_extension(format!("ready.tmp.{}.{}", std::process::id(), sequence));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    let (temporary_path, file) = temporary
        .ok_or_else(|| "could not reserve a ready-world cache temporary file".to_owned())?;
    let result = (|| {
        let mut writer = GzEncoder::new(BufWriter::new(file), Compression::fast());
        state
            .write_ready_world_snapshot_to(&mut writer)
            .map_err(|error| error.to_string())?;
        let mut writer = writer.finish().map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        let file = writer.into_inner().map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        let bytes = file.metadata().map_err(|error| error.to_string())?.len();
        drop(file);
        if path.exists() {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        }
        fs::rename(&temporary_path, path).map_err(|error| error.to_string())?;
        Ok(bytes)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

fn startup_scheduler_limits() -> SchedulerDrainLimits {
    let defaults = SchedulerDrainLimits::default();
    let max_rounds = env::var_os("DREAM64_STARTUP_MAX_ROUNDS")
        .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
        .unwrap_or(defaults.max_rounds);
    let max_ticks = env::var_os("DREAM64_STARTUP_MAX_TICKS")
        .and_then(|value| value.to_str().and_then(|value| value.parse::<u64>().ok()))
        .unwrap_or(defaults.max_ticks);
    SchedulerDrainLimits {
        max_ticks,
        max_rounds,
    }
}

fn run_lobby_preflight(
    environment: &Path,
    world_params: &str,
    index: &LifecycleIndex,
    runtime: &mut RuntimeImage,
    executable: &ExecutableProcedures,
    serve: bool,
) -> ExitCode {
    const MAX_TICKS: u64 = 1_000;
    let started = Instant::now();
    let world_path = TypePath::parse("/world").expect("built-in world path is valid");
    let world = match runtime
        .canonical_world()
        .map(Ok)
        .unwrap_or_else(|| runtime.allocate_datum(&world_path))
    {
        Ok(world) => world,
        Err(error) => {
            eprintln!("lobby-preflight: world allocation failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let genesis = match index
        .find_path("/world")
        .map(|lifecycle| lifecycle.targets.get(LifecycleKind::Genesis))
    {
        Some(LifecycleResolution::Resolved(target)) => Some(target),
        Some(LifecycleResolution::Absent) | None => None,
        Some(LifecycleResolution::Unsupported(issue)) => {
            eprintln!(
                "lobby-preflight: world Genesis is unsupported: {}",
                issue.message
            );
            return ExitCode::FAILURE;
        }
    };
    let mut state = runtime.take_execution_state();
    let world_params = match state.decode_params_list(world_params) {
        Ok(params) => params,
        Err(error) => {
            eprintln!("lobby-preflight: world params decoding failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = state.heap_mut().set_datum_field(
        world,
        FieldName::parse("params").expect("built-in world params field is valid"),
        world_params,
    ) {
        eprintln!("lobby-preflight: world params assignment failed: {error}");
        return ExitCode::FAILURE;
    }
    state.set_global(
        FieldName::parse("world").expect("built-in world global is valid"),
        Value::Datum(world),
    );
    if let Some(target) = genesis {
        let Some(entry) = executable.implementation(target.implementation) else {
            eprintln!("lobby-preflight: world Genesis VM target is missing");
            return ExitCode::FAILURE;
        };
        if let Err(error) = dm_vm::execute_module_in_context(
            executable.module(),
            entry,
            &[],
            &mut state,
            &dm_vm::ExecutionContext::new(Value::Datum(world), Value::Null),
        ) {
            eprintln!("lobby-preflight: world Genesis failed: {error}");
            return ExitCode::FAILURE;
        }
        for tick in 0..=MAX_TICKS {
            if state.scheduled_task_count() == 0 {
                break;
            }
            if let Err(error) = advance_scheduler(
                executable.module(),
                u64::from(tick != 0),
                ExecutionLimits::default(),
                &mut state,
            ) {
                eprintln!("lobby-preflight: world Genesis continuation failed: {error}");
                return ExitCode::FAILURE;
            }
            if tick == MAX_TICKS && state.scheduled_task_count() != 0 {
                eprintln!("lobby-preflight: world Genesis exceeded {MAX_TICKS} ticks");
                return ExitCode::FAILURE;
            }
        }
    }
    let world_new = match index
        .find_path("/world")
        .map(|lifecycle| lifecycle.targets.get(LifecycleKind::New))
    {
        Some(LifecycleResolution::Resolved(target)) => target,
        Some(LifecycleResolution::Absent) | None => {
            eprintln!("lobby-preflight: world/New is absent");
            return ExitCode::FAILURE;
        }
        Some(LifecycleResolution::Unsupported(issue)) => {
            eprintln!(
                "lobby-preflight: world/New is unsupported: {}",
                issue.message
            );
            return ExitCode::FAILURE;
        }
    };
    let Some(entry) = executable.implementation(world_new.implementation) else {
        eprintln!("lobby-preflight: world/New VM target is missing");
        return ExitCode::FAILURE;
    };
    if let Err(error) = dm_vm::execute_module_in_context(
        executable.module(),
        entry,
        &[],
        &mut state,
        &dm_vm::ExecutionContext::new(Value::Datum(world), Value::Null),
    ) {
        eprintln!("lobby-preflight: world/New failed: {error}");
        return ExitCode::FAILURE;
    }
    for tick in 0..=MAX_TICKS {
        if state.scheduled_task_count() == 0 {
            break;
        }
        if let Err(error) = advance_scheduler(
            executable.module(),
            u64::from(tick != 0),
            ExecutionLimits::default(),
            &mut state,
        ) {
            eprintln!("lobby-preflight: world/New continuation failed: {error}");
            return ExitCode::FAILURE;
        }
        if tick == MAX_TICKS && state.scheduled_task_count() != 0 {
            eprintln!("lobby-preflight: world/New exceeded {MAX_TICKS} ticks");
            return ExitCode::FAILURE;
        }
    }
    let area = match state
        .allocate_compact_map_datum(TypePath::parse("/area").expect("built-in area path is valid"))
    {
        Ok(area) => area,
        Err(error) => {
            eprintln!("lobby-preflight: minimal area allocation failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let turf = match state
        .allocate_compact_map_datum(TypePath::parse("/turf").expect("built-in turf path is valid"))
    {
        Ok(turf) => turf,
        Err(error) => {
            eprintln!("lobby-preflight: minimal turf allocation failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    for (name, value) in [("x", 1), ("y", 1), ("z", 1)] {
        if let Err(error) = state.heap_mut().set_datum_field(
            turf,
            FieldName::parse(name).expect("built-in coordinate field is valid"),
            Value::number(value as f32),
        ) {
            eprintln!("lobby-preflight: minimal turf coordinate failed: {error}");
            return ExitCode::FAILURE;
        }
    }
    if let Err(error) = state.heap_mut().set_datum_field(
        turf,
        FieldName::parse("loc").expect("built-in loc field is valid"),
        Value::Datum(area),
    ) {
        eprintln!("lobby-preflight: minimal turf area assignment failed: {error}");
        return ExitCode::FAILURE;
    }
    state.rebuild_world_geometry();
    if serve {
        let ipc_address =
            env::var("DREAM64_IPC_ADDR").unwrap_or_else(|_| "0.0.0.0:51664".to_owned());
        let mut ipc = match parse_loopback_address(&ipc_address).and_then(LoopbackIpc::bind) {
            Ok(ipc) => ipc,
            Err(error) => {
                eprintln!("lobby-preview: loopback IPC: {error}");
                return ExitCode::FAILURE;
            }
        };
        report_public_endpoint(ipc.local_addr().port());
        eprintln!(
            "lobby-preview: ready project={} loopback-ipc={} hidden_guests=0 elapsed_ms={}",
            environment.display(),
            ipc.local_addr(),
            started.elapsed().as_millis()
        );
        let mut slices = 0u64;
        loop {
            let slice_started = Instant::now();
            ipc.apply_executable_tick_boundary(executable, &mut state);
            if let Err(error) = advance_scheduler(
                executable.module(),
                1,
                ExecutionLimits::default(),
                &mut state,
            ) {
                eprintln!(
                    "lobby-preview: scheduler failed: {}\ncall stack: {:#?}",
                    error.message, error.call_stack
                );
                return ExitCode::FAILURE;
            }
            slices = slices.saturating_add(1);
            if slices == 1 || slices % 100 == 0 {
                eprintln!(
                    "lobby-preview: scheduler slice={} tick={} pending={}",
                    slices,
                    state.scheduler_tick(),
                    state.scheduled_task_count()
                );
            }
            let tick = std::time::Duration::from_millis(100);
            if let Some(remaining) = tick.checked_sub(slice_started.elapsed()) {
                std::thread::sleep(remaining);
            }
        }
    }
    let attached = match state.connect_local_guest(executable.module()) {
        Ok(attached) => attached,
        Err(error) => {
            eprintln!("lobby-preflight: guest attach failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let attached_mob_type = state
        .heap()
        .datum(attached.mob)
        .map(|datum| datum.type_path().to_string())
        .unwrap_or_else(|_| "<stale>".to_owned());
    eprintln!("lobby-preflight: attached mob_type={attached_mob_type}");
    let skin_controls = state.client_session(attached.client).map_or(0, |session| {
        session
            .ui()
            .tree()
            .windows
            .iter()
            .map(|window| window.controls.len())
            .sum()
    });
    let mut events = Vec::new();
    for tick in 0..=MAX_TICKS {
        if let Err(error) = advance_scheduler(
            executable.module(),
            u64::from(tick != 0),
            ExecutionLimits::default(),
            &mut state,
        ) {
            let bound_mob = state
                .heap()
                .datum_field(
                    attached.client,
                    &FieldName::parse("mob").expect("client mob field"),
                )
                .cloned()
                .unwrap_or(Value::Null);
            let hud = state
                .heap()
                .datum_field(
                    attached.mob,
                    &FieldName::parse("hud_used").expect("mob hud field"),
                )
                .cloned()
                .unwrap_or(Value::Null);
            eprintln!(
                "lobby-preflight: /client/New failed tick={tick} skin_controls={skin_controls} ui_events={} elapsed_ms={} bound_mob={} hud_used={} error={}\ncall stack: {:#?}",
                events.len(),
                started.elapsed().as_millis(),
                bound_mob,
                hud,
                error.message,
                error.call_stack
            );
            return ExitCode::FAILURE;
        }
        events.extend(state.take_local_client_outbound_events(attached.client));
        if state.scheduled_task_count() == 0 {
            if events.is_empty() {
                eprintln!(
                    "lobby-preflight: /client/New completed without UI events ticks={tick} skin_controls={skin_controls} elapsed_ms={}",
                    started.elapsed().as_millis()
                );
                return ExitCode::FAILURE;
            }
            eprintln!(
                "lobby-preflight: ready project={} ticks={} skin_controls={} ui_events={} elapsed_ms={}",
                environment.display(),
                tick,
                skin_controls,
                events.len(),
                started.elapsed().as_millis()
            );
            return ExitCode::SUCCESS;
        }
    }
    eprintln!(
        "lobby-preflight: /client/New did not complete within {MAX_TICKS} ticks pending={} ui_events={} elapsed_ms={}",
        state.scheduled_task_count(),
        events.len(),
        started.elapsed().as_millis()
    );
    ExitCode::FAILURE
}

const COMPILATION_ARTIFACT_SECTION: u32 = 1;
const EXECUTABLE_ARTIFACT_SECTION: u32 = 2;
const RUNTIME_STRUCTURAL_ARTIFACT_SECTION: u32 = 3;
const COMPACT_WORDCODE_ARTIFACT_SECTION: u32 = 4;
const STABLE_ID_ARTIFACT_SECTION: u32 = 5;
const RUNTIME_LINKED_ARTIFACT_SECTION: u32 = 6;
const LIFECYCLE_DIRECTORY_ARTIFACT_SECTION: u32 = 7;
const DEFAULT_MAP_ARTIFACT_SECTION: u32 = 8;
const BOOT_MANIFEST_ARTIFACT_SECTION: u32 = 9;
const DMM_MEASUREMENT_ARTIFACT_SECTION: u32 = 10;
const PARSED_DMM_ARTIFACT_SECTION: u32 = 11;
const PROCEDURE_SEMANTICS_ARTIFACT_SECTION: u32 = 12;

struct PreparedStandaloneRuntime {
    executable: ExecutableProcedures,
    runtime: RuntimeImage,
    lifecycle: LifecycleIndex,
    map_path: String,
    world: dm_world::WorldPlan,
    readiness: Option<HeadlessReadinessProbe>,
    dmm_measurements: BTreeMap<String, dm_lifecycle::PortableDmmMeasurement>,
    _parsed_dmm_cache: BTreeMap<String, dm_lifecycle::PortableParsedDmm>,
}

fn prepare_standalone_runtime(path: &Path) -> Result<PreparedStandaloneRuntime, String> {
    let fingerprint =
        CompiledArtifact::peek_project_fingerprint(path).map_err(|error| error.to_string())?;
    let artifact =
        CompiledArtifact::read_from(path, fingerprint).map_err(|error| error.to_string())?;
    let executable_payload = artifact
        .section(EXECUTABLE_ARTIFACT_SECTION)
        .ok_or("standalone artifact is missing executable section")?
        .payload();
    let mut executable = ExecutableProcedures::decode_compiled_artifact(executable_payload)?;
    decode_and_attach_procedure_semantics(
        artifact
            .section(PROCEDURE_SEMANTICS_ARTIFACT_SECTION)
            .ok_or("standalone artifact is missing procedure semantic directory")?
            .payload(),
        executable.module_mut(),
    )?;
    if let Some(selector) = env::var_os("DREAM64_SEMANTIC_DIGEST_PATH") {
        let selector = selector.to_string_lossy();
        if selector.len() > 4096 || !selector.starts_with('/') {
            return Err("DREAM64_SEMANTIC_DIGEST_PATH is invalid".to_owned());
        }
        let procedure = executable
            .module()
            .effective_procedure_id(&selector)
            .or_else(|| executable.module().procedure_id(&selector))
            .ok_or_else(|| format!("semantic digest procedure {selector:?} is absent"))?;
        let digest = executable
            .module()
            .procedure_semantic_digest(procedure)
            .ok_or("semantic digest was not attached")?;
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        eprintln!(
            "boot-progress: procedure-semantic path={} id={} sha256={hex}",
            selector,
            procedure.index(),
        );
    }
    if let Some(section) = artifact.section(COMPACT_WORDCODE_ARTIFACT_SECTION) {
        let compact = dm_vm::CompactWordcodeImage::decode(section.payload())
            .map_err(|error| error.to_string())?;
        executable
            .module_mut()
            .attach_compact_wordcode(compact)
            .map_err(|error| error.to_string())?;
    }
    let runtime = RuntimeImage::decode_linked_artifact(
        artifact
            .section(RUNTIME_LINKED_ARTIFACT_SECTION)
            .ok_or("standalone artifact is missing linked runtime section")?
            .payload(),
        executable.module(),
    )?;
    let lifecycle = LifecycleIndex::decode_portable(
        artifact
            .section(LIFECYCLE_DIRECTORY_ARTIFACT_SECTION)
            .ok_or("standalone artifact is missing lifecycle directory")?
            .payload(),
    )?;
    let (map_path, world) = dm_world::decode_named_portable_plan(
        artifact
            .section(DEFAULT_MAP_ARTIFACT_SECTION)
            .ok_or("standalone artifact is missing default map directory")?
            .payload(),
    )?;
    let readiness = artifact
        .section(BOOT_MANIFEST_ARTIFACT_SECTION)
        .map(|section| HeadlessReadinessProbe::decode_portable_manifest(section.payload()))
        .transpose()
        .map_err(|error| format!("boot manifest: {error}"))?;
    let dmm_measurements = decode_dmm_measurements(
        artifact
            .section(DMM_MEASUREMENT_ARTIFACT_SECTION)
            .ok_or("standalone artifact is missing DMM measurement catalog")?
            .payload(),
    )?;
    let parsed_dmm_cache = decode_parsed_dmm_cache(
        artifact
            .section(PARSED_DMM_ARTIFACT_SECTION)
            .ok_or("standalone artifact is missing parsed DMM catalog")?
            .payload(),
    )?;
    let has_ticker = runtime
        .types()
        .any(|(path, _)| path.as_str() == "/datum/controller/subsystem/ticker");
    if has_ticker && readiness.is_none() {
        return Err("standalone ticker runtime is missing its boot readiness manifest".to_owned());
    }
    Ok(PreparedStandaloneRuntime {
        executable,
        runtime,
        lifecycle,
        map_path,
        world,
        readiness,
        dmm_measurements,
        _parsed_dmm_cache: parsed_dmm_cache,
    })
}

fn run_standalone_linked_boot(
    path: &Path,
    audit: bool,
    startup_ipc: Option<LoopbackIpc>,
    process_started: Instant,
) -> ExitCode {
    let PreparedStandaloneRuntime {
        executable,
        mut runtime,
        lifecycle,
        map_path,
        world,
        readiness,
        dmm_measurements,
        _parsed_dmm_cache: parsed_dmm_cache,
    } = match prepare_standalone_runtime(path) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    runtime.set_dmm_measurements(Arc::new(
        dmm_measurements
            .into_iter()
            .map(|(path, entry)| (path, entry.measurement))
            .collect(),
    ));
    runtime.set_parsed_dmm_cache(Arc::new(
        parsed_dmm_cache
            .into_iter()
            .map(|(path, entry)| {
                (
                    path,
                    dm_vm::ParsedDmm {
                        digest: entry.digest,
                        tgm: entry.tgm,
                        key_len: entry.key_len,
                        line_len: entry.line_len,
                        bounds: entry.bounds,
                        models: entry.models,
                        grids: entry
                            .grids
                            .into_iter()
                            .map(|grid| dm_vm::ParsedDmmGrid {
                                x: grid.x,
                                y: grid.y,
                                z: grid.z,
                                lines: grid.lines,
                            })
                            .collect(),
                    },
                )
            })
            .collect(),
    ));
    let plan = build_initialization_plan(&runtime, &lifecycle, &world, map_path.clone());
    if let Err(errors) = runtime.preflight_instance_initializers(
        world
            .templates()
            .values()
            .flat_map(|template| template.initializers.iter())
            .filter_map(|initializer| TypePath::parse(&initializer.path).ok()),
    ) {
        eprintln!("initializer preflight failed: {} error(s)", errors.len());
        return ExitCode::FAILURE;
    }
    let allocation = match allocate_world(&world, &mut runtime) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("world allocation: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut precompiled = precompile_portable_lifecycle_for_world(&lifecycle, &world, executable);
    let execution = if audit {
        audit_initialization_plan_with_precompiled(
            &lifecycle,
            &plan,
            &allocation,
            &mut runtime,
            startup_scheduler_limits(),
            &mut precompiled,
        )
    } else {
        execute_boot_initialization_plan_with_precompiled(
            &lifecycle,
            &plan,
            &allocation,
            &mut runtime,
            startup_scheduler_limits(),
            readiness.as_ref(),
            &mut precompiled,
        )
    };
    let execution = match execution {
        Ok(value) => value,
        Err(error) => {
            eprintln!("initialization: {error}");
            return ExitCode::FAILURE;
        }
    };
    print_boot_summary(&allocation, &execution);
    if audit {
        eprintln!(
            "boot-progress: standalone linked audit complete elapsed_ms={}",
            process_started.elapsed().as_millis()
        );
        return ExitCode::SUCCESS;
    }
    eprintln!(
        "boot-progress: standalone linked runtime ready elapsed_ms={}",
        process_started.elapsed().as_millis()
    );
    run_persistent_server_loop(
        &mut runtime,
        &mut precompiled,
        startup_ipc,
        readiness.as_ref(),
    )
}

#[derive(Clone, Copy)]
struct PreparedCacheStats {
    project_snapshot_hit: bool,
    parsed_syntax_hit: Option<bool>,
    artifact_hit: bool,
}

struct PreparedCompiledExecutable {
    compilation: Compilation,
    procedures: ProcedureRegistry,
    executable: ExecutableProcedures,
    structural_seed: RuntimeStructuralSeed,
    linked_runtime: Option<RuntimeImage>,
    lifecycle_index: Option<LifecycleIndex>,
    default_map: Option<(String, dm_world::WorldPlan)>,
    project_snapshot_hit: bool,
    parsed_syntax_hit: Option<bool>,
    artifact_hit: bool,
    miss_reason: Option<String>,
    new_lowerings: usize,
}

fn prepare_standalone_artifact(artifact_file: &Path) -> Result<PreparedCompiledExecutable, String> {
    let project_fingerprint = CompiledArtifact::peek_project_fingerprint(artifact_file)
        .map_err(|error| error.to_string())?;
    let (compilation, executable, structural_seed, linked_runtime, lifecycle_index, default_map) =
        decode_compiled_executable(artifact_file, project_fingerprint)?;
    let procedures = ProcedureRegistry::build_lazy(&compilation);
    Ok(PreparedCompiledExecutable {
        compilation,
        procedures,
        executable,
        structural_seed,
        linked_runtime,
        lifecycle_index,
        default_map,
        project_snapshot_hit: false,
        parsed_syntax_hit: None,
        artifact_hit: true,
        miss_reason: None,
        new_lowerings: 0,
    })
}

fn prepare_compiled_executable(
    environment: &Path,
    cache_file: &Path,
    artifact_file: &Path,
    allow_compile: bool,
) -> Result<PreparedCompiledExecutable, String> {
    // The compact project snapshot validates every discovered input before we
    // trust the heavyweight executable. Normal interactive boots use the
    // sidecar's path/length/high-resolution-mtime manifest, avoiding a second
    // read of the entire (often multi-gigabyte) project tree. Reproducible
    // release verification can request the adversarially strict byte-for-byte
    // pass with DREAM64_STRICT_SOURCE_HASH=1.
    let phase = Instant::now();
    let strict_source_hash = env::var_os("DREAM64_STRICT_SOURCE_HASH")
        .is_some_and(|value| !matches!(value.to_string_lossy().trim(), "" | "0" | "false" | "no"));
    eprintln!(
        "compile-cache-phase: project-source-validation-start mode={} project={}",
        if strict_source_hash {
            "strict-bytes"
        } else {
            "metadata"
        },
        environment.display(),
    );
    let (project, project_snapshot_hit, project_fingerprint) = if strict_source_hash {
        Project::load_cached_exact_with_fingerprint(environment, cache_file)
            .map_err(|error| format!("project snapshot: {error}"))?
    } else {
        let (project, hit) = Project::load_cached(environment, cache_file)
            .map_err(|error| format!("project snapshot: {error}"))?;
        let fingerprint = project.content_fingerprint();
        (project, hit, fingerprint)
    };
    eprintln!(
        "compile-cache-phase: project-source-validation mode={} elapsed_ms={} hit={project_snapshot_hit}",
        if strict_source_hash {
            "strict-bytes"
        } else {
            "metadata"
        },
        phase.elapsed().as_millis(),
    );
    let project_fingerprint = *project_fingerprint.as_bytes();
    let miss_reason = if project_snapshot_hit {
        match decode_compiled_executable(artifact_file, project_fingerprint) {
            Ok((
                compilation,
                executable,
                structural_seed,
                linked_runtime,
                lifecycle_index,
                default_map,
            )) => {
                let phase = Instant::now();
                let procedures = ProcedureRegistry::build_lazy(&compilation);
                eprintln!(
                    "compile-cache-phase: procedure-registry elapsed_ms={}",
                    phase.elapsed().as_millis()
                );
                return Ok(PreparedCompiledExecutable {
                    compilation,
                    procedures,
                    executable,
                    structural_seed,
                    linked_runtime,
                    lifecycle_index,
                    default_map,
                    project_snapshot_hit,
                    parsed_syntax_hit: None,
                    artifact_hit: true,
                    miss_reason: None,
                    new_lowerings: 0,
                });
            }
            Err(error) => error,
        }
    } else {
        "project snapshot changed or was not cached".to_owned()
    };
    if !allow_compile {
        return Err(format!(
            "runtime artifacts are unavailable or stale: {miss_reason}; run `dream64-compiler {}` before booting",
            environment.display(),
        ));
    }

    // One miss performs exactly one ordinary frontend compilation followed by
    // one complete symbolic link and deterministic eager materialization.
    let phase = Instant::now();
    let compiler_database_file = compiler_database_file(cache_file);
    let (compilation, cache_stats) = CompilerDatabase::new()
        .compile_persistent_prevalidated(
            project,
            project_snapshot_hit,
            &compiler_database_file,
            BuildMode::Incremental,
            persistent_semantic_digest(),
            persistent_build_configuration_digest(),
        )
        .map_err(|error| error.to_string())?;
    eprintln!(
        "compile-cache-phase: persistent-frontend elapsed_ms={} linked_cache={} project_snapshot_hit={} parsed_syntax_hit={} syntax_reused={} syntax_reparsed={} changed_inputs={} invalidated_sections={} database={}",
        phase.elapsed().as_millis(),
        if cache_stats.linked_sections_reused != 0 {
            "hit"
        } else {
            "miss"
        },
        cache_stats.project_snapshot_hit,
        cache_stats.parsed_syntax_hit,
        cache_stats.syntax_files_reused,
        cache_stats.syntax_files_reparsed,
        cache_stats.changed_inputs,
        cache_stats.invalidated_sections,
        compiler_database_file.display(),
    );
    let mut persistent_database = PersistentCompilerDatabase::read(&compiler_database_file)
        .map_err(|error| format!("read persistent compiler database: {error}"))?;
    let procedure_ids = stable_ids_for_namespace(&persistent_database, "procedure");
    let type_ids = stable_ids_for_namespace(&persistent_database, "type");
    validate_procedure_id_inventory(&compilation, &procedure_ids)?;
    let phase = Instant::now();
    let procedures = ProcedureRegistry::build_with_stable_ids(&compilation, &procedure_ids)
        .map_err(|error| format!("stable procedure linking: {error}"))?;
    let procedure_digest = procedures.persistent_semantic_digest(&compilation);
    eprintln!(
        "compile-cache-phase: procedure-registry elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let phase = Instant::now();
    let cached_payload = persistent_database
        .sections
        .iter()
        .find(|section| section.section_id == PERSISTENT_EXECUTABLE_SECTION)
        .filter(|section| section.content_digest == procedure_digest)
        .and_then(|_| persistent_database.section_payload(PERSISTENT_EXECUTABLE_SECTION));
    let (mut executable, executable_cache_hit) = cached_payload
        .as_deref()
        .and_then(|payload| ExecutableProcedures::decode_compiled_artifact(payload).ok())
        .map_or_else(
            || {
                procedures
                    .compile_vm_all_symbolic_deferred(&compilation)
                    .map_err(|error| format!("complete executable lowering: {error}"))?
                    .into_fully_eager_bounded(MAX_EAGER_ARTIFACT_DIAGNOSTICS)
                    .map(|executable| (executable, false))
                    .map_err(|error| format!("complete executable lowering: {error}"))
            },
            |executable| Ok((executable, true)),
        )?;
    eprintln!(
        "compile-cache-phase: executable-link cache={} elapsed_ms={}",
        if executable_cache_hit { "hit" } else { "miss" },
        phase.elapsed().as_millis()
    );
    if executable.module().deferred_procedure_count() != 0 {
        return Err("complete executable lowering retained deferred procedures".to_owned());
    }
    let new_lowerings = if executable_cache_hit {
        0
    } else {
        executable.module().procedure_count()
    };
    let phase = Instant::now();
    let compilation_payload = compilation.encode_compiled_artifact();
    eprintln!(
        "compile-cache-phase: frontend-encode elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let phase = Instant::now();
    let executable_payload = executable.encode_compiled_artifact()?;
    let procedure_semantics_payload = encode_procedure_semantics(executable.module())?;
    eprintln!(
        "compile-cache-phase: executable-encode elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    if !executable_cache_hit {
        let pages = persistent_database
            .replace_paged_section(
                PERSISTENT_EXECUTABLE_SECTION,
                PERSISTENT_EXECUTABLE_PAGE_BASE,
                procedure_digest,
                &executable_payload,
                vec![1],
            )
            .map_err(|error| format!("cache executable link: {error}"))?;
        persistent_database
            .write_atomic(&compiler_database_file)
            .map_err(|error| format!("write executable compiler database: {error}"))?;
        eprintln!(
            "compile-cache-phase: executable-cache-write bytes={} pages={pages}",
            executable_payload.len()
        );
    }
    let phase = Instant::now();
    let structural_seed = RuntimeStructuralSeed::build_with_stable_ids(&compilation, &type_ids)
        .map_err(|error| format!("runtime structural seed: {error}"))?;
    let structural_payload = structural_seed.encode_compiled_artifact();
    eprintln!(
        "compile-cache-phase: runtime-structural-encode elapsed_ms={} bytes={}",
        phase.elapsed().as_millis(),
        structural_payload.len(),
    );
    let stable_id_payload = encode_stable_id_table(&persistent_database.stable_ids)
        .map_err(|error| format!("stable-ID runtime section: {error}"))?;
    let mut linked_runtime = RuntimeImage::from_compilation(&compilation)
        .map_err(|error| format!("linked runtime image: {error}"))?;
    let boot_manifest_payload = lobby_pregame_readiness(&compilation, &linked_runtime)
        .map(|probe| probe.encode_portable_manifest())
        .transpose()
        .map_err(|error| format!("boot manifest: {error}"))?;
    let parsed_dmm_cache = build_parsed_dmm_cache(&compilation)?;
    let dmm_measurements = dmm_measurements_from_parsed(&parsed_dmm_cache);
    let dmm_measurement_payload = encode_dmm_measurements(&dmm_measurements)?;
    let parsed_dmm_payload = encode_parsed_dmm_cache(&parsed_dmm_cache)?;
    eprintln!(
        "compile-progress: dmm-measurements entries={} bytes={}",
        dmm_measurements.len(),
        dmm_measurement_payload.len(),
    );
    linked_runtime
        .materialize_linked_artifact_initializers(MAX_EAGER_ARTIFACT_DIAGNOSTICS)
        .map_err(|error| format!("linked runtime initializer module: {error}"))?;
    let linked_runtime_payload = linked_runtime
        .encode_linked_artifact(executable.module())
        .map_err(|error| format!("linked runtime image: {error}"))?;
    let lifecycle_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
    let lifecycle_directory_payload = lifecycle_index
        .encode_portable()
        .map_err(|error| format!("lifecycle directory: {error}"))?;
    let default_map_payload = load_map(&compilation, None)
        .ok()
        .map(|(path, source)| {
            let map =
                dm_map::parse(&source).map_err(|error| format!("default map {path}: {error}"))?;
            let plan = dm_world::build_plan(&map, &compilation);
            dm_world::encode_named_portable_plan(&path, &plan)
                .map_err(|error| format!("default map directory: {error}"))
        })
        .transpose()?;
    let prepared_default_map = default_map_payload
        .as_deref()
        .map(dm_world::decode_named_portable_plan)
        .transpose()?;
    eprintln!(
        "compile-progress: executable-artifact-payloads frontend_bytes={} executable_bytes={}",
        compilation_payload.len(),
        executable_payload.len(),
    );
    let fingerprint = *compilation.project().content_fingerprint().as_bytes();
    let compact_payload = if env::var_os("DREAM64_DISABLE_COMPACT_WORDCODE").is_none() {
        let compact = dm_vm::CompactWordcodeImage::build(executable.module())
            .map_err(|error| format!("compact wordcode: {error}"))?;
        let payload = compact
            .encode()
            .map_err(|error| format!("compact wordcode: {error}"))?;
        eprintln!(
            "compile-cache-phase: compact-wordcode-encode bytes={} strings={} procedures={} words={} specialized={}",
            payload.len(),
            compact.string_count(),
            compact.procedure_count(),
            compact.word_count(),
            compact.specialized_word_count(),
        );
        executable
            .module_mut()
            .attach_compact_wordcode(compact)
            .map_err(|error| format!("compact wordcode: {error}"))?;
        Some(payload)
    } else {
        None
    };
    let phase = Instant::now();
    let mut sections = vec![
        ArtifactSection::new(COMPILATION_ARTIFACT_SECTION, compilation_payload),
        ArtifactSection::new(EXECUTABLE_ARTIFACT_SECTION, executable_payload),
        ArtifactSection::new(RUNTIME_STRUCTURAL_ARTIFACT_SECTION, structural_payload),
    ];
    if let Some(payload) = compact_payload {
        sections.push(ArtifactSection::new(
            COMPACT_WORDCODE_ARTIFACT_SECTION,
            payload,
        ));
    }
    sections.push(ArtifactSection::new(
        STABLE_ID_ARTIFACT_SECTION,
        stable_id_payload,
    ));
    sections.push(ArtifactSection::new(
        RUNTIME_LINKED_ARTIFACT_SECTION,
        linked_runtime_payload,
    ));
    sections.push(ArtifactSection::new(
        LIFECYCLE_DIRECTORY_ARTIFACT_SECTION,
        lifecycle_directory_payload,
    ));
    if let Some(payload) = default_map_payload {
        sections.push(ArtifactSection::new(DEFAULT_MAP_ARTIFACT_SECTION, payload));
    }
    if let Some(payload) = boot_manifest_payload {
        sections.push(ArtifactSection::new(
            BOOT_MANIFEST_ARTIFACT_SECTION,
            payload,
        ));
    }
    sections.push(ArtifactSection::new(
        DMM_MEASUREMENT_ARTIFACT_SECTION,
        dmm_measurement_payload,
    ));
    sections.push(ArtifactSection::new(
        PARSED_DMM_ARTIFACT_SECTION,
        parsed_dmm_payload,
    ));
    sections.push(ArtifactSection::new(
        PROCEDURE_SEMANTICS_ARTIFACT_SECTION,
        procedure_semantics_payload,
    ));
    let runtime_artifact = CompiledArtifact::new(fingerprint, sections)
        .map_err(|error| format!("build executable artifact: {error}"))?;
    eprintln!(
        "compile-cache-phase: envelope-build elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let phase = Instant::now();
    let runtime_write_stats = runtime_artifact
        .write_atomic_with_stats(artifact_file)
        .map_err(|error| format!("write runtime artifact: {error}"))?;
    eprintln!(
        "compile-cache-phase: artifact-write elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    eprintln!(
        "compile-progress: runtime-artifact-write runtime={} runtime_bytes={} payload_bytes={} peak_staging_bytes={} write_calls={}",
        artifact_file.display(),
        runtime_write_stats.encoded_bytes,
        runtime_write_stats.payload_bytes,
        runtime_write_stats.peak_staging_bytes,
        runtime_write_stats.write_calls,
    );
    Ok(PreparedCompiledExecutable {
        compilation,
        procedures,
        executable,
        structural_seed,
        linked_runtime: Some(linked_runtime),
        lifecycle_index: Some(lifecycle_index),
        default_map: prepared_default_map,
        project_snapshot_hit,
        parsed_syntax_hit: Some(cache_stats.parsed_syntax_hit),
        artifact_hit: false,
        miss_reason: Some(miss_reason),
        new_lowerings,
    })
}

fn decode_compiled_executable(
    artifact_file: &Path,
    project_fingerprint: [u8; 16],
) -> Result<
    (
        Compilation,
        ExecutableProcedures,
        RuntimeStructuralSeed,
        Option<RuntimeImage>,
        Option<LifecycleIndex>,
        Option<(String, dm_world::WorldPlan)>,
    ),
    String,
> {
    let phase = Instant::now();
    let runtime_artifact = match CompiledArtifact::read_from(artifact_file, project_fingerprint) {
        Ok(artifact) => artifact,
        Err(error) => {
            eprintln!(
                "compile-cache-phase: runtime-artifact-read-rejected elapsed_ms={} reason={error}",
                phase.elapsed().as_millis()
            );
            return Err(error.to_string());
        }
    };
    eprintln!(
        "compile-cache-phase: artifact-read-checksum elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let runtime_storage = runtime_artifact.storage_stats();
    eprintln!(
        "boot-progress: runtime-artifact-storage payload_bytes={} backing_bytes={} backing_allocations={}",
        runtime_storage.payload_bytes,
        runtime_storage.backing_bytes,
        runtime_storage.backing_allocations,
    );
    if !(3..=12).contains(&runtime_artifact.sections().len()) {
        return Err(format!(
            "runtime artifact contains {} sections instead of 3 through 5",
            runtime_artifact.sections().len()
        ));
    }
    if runtime_artifact.sections().iter().any(|section| {
        !matches!(
            section.id(),
            COMPILATION_ARTIFACT_SECTION
                | EXECUTABLE_ARTIFACT_SECTION
                | RUNTIME_STRUCTURAL_ARTIFACT_SECTION
                | COMPACT_WORDCODE_ARTIFACT_SECTION
                | STABLE_ID_ARTIFACT_SECTION
                | RUNTIME_LINKED_ARTIFACT_SECTION
                | LIFECYCLE_DIRECTORY_ARTIFACT_SECTION
                | DEFAULT_MAP_ARTIFACT_SECTION
                | BOOT_MANIFEST_ARTIFACT_SECTION
                | DMM_MEASUREMENT_ARTIFACT_SECTION
                | PARSED_DMM_ARTIFACT_SECTION
                | PROCEDURE_SEMANTICS_ARTIFACT_SECTION
        )
    }) {
        return Err("runtime artifact contains an unknown section".to_owned());
    }
    let frontend_payload = runtime_artifact
        .section(COMPILATION_ARTIFACT_SECTION)
        .ok_or_else(|| "runtime artifact is missing the bootstrap section".to_owned())?
        .payload();
    let executable_payload = runtime_artifact
        .section(EXECUTABLE_ARTIFACT_SECTION)
        .ok_or_else(|| "runtime artifact is missing the bytecode section".to_owned())?
        .payload();
    let structural_payload = runtime_artifact
        .section(RUNTIME_STRUCTURAL_ARTIFACT_SECTION)
        .ok_or_else(|| "runtime artifact is missing the structural section".to_owned())?
        .payload();
    let compact_payload = runtime_artifact
        .section(COMPACT_WORDCODE_ARTIFACT_SECTION)
        .map(ArtifactSection::payload);
    let stable_ids = if let Some(payload) = runtime_artifact
        .section(STABLE_ID_ARTIFACT_SECTION)
        .map(ArtifactSection::payload)
    {
        let stable_ids = decode_stable_id_table(payload)
            .map_err(|error| format!("runtime artifact stable-ID section: {error}"))?;
        eprintln!("boot-progress: stable-ids validated={}", stable_ids.len());
        Some(stable_ids)
    } else {
        None
    };
    let parallel_phase = Instant::now();
    let (compilation, mut executable, structural_seed, compact) = std::thread::scope(|scope| {
        let frontend = scope.spawn(|| {
            let started = Instant::now();
            (
                Compilation::decode_compiled_artifact(frontend_payload),
                started.elapsed(),
            )
        });
        let bytecode = scope.spawn(|| {
            let started = Instant::now();
            (
                ExecutableProcedures::decode_compiled_artifact(executable_payload),
                started.elapsed(),
            )
        });
        let structural = scope.spawn(|| {
            let started = Instant::now();
            (
                RuntimeStructuralSeed::decode_compiled_artifact(structural_payload),
                started.elapsed(),
            )
        });
        let compact = compact_payload.map(|payload| {
            scope.spawn(|| {
                let started = Instant::now();
                (
                    dm_vm::CompactWordcodeImage::decode(payload),
                    started.elapsed(),
                )
            })
        });
        let (compilation, frontend_elapsed) = frontend
            .join()
            .map_err(|_| "frontend artifact decoder panicked".to_owned())?;
        let (executable, executable_elapsed) = bytecode
            .join()
            .map_err(|_| "bytecode artifact decoder panicked".to_owned())?;
        let (structural_seed, structural_elapsed) = structural
            .join()
            .map_err(|_| "runtime structural artifact decoder panicked".to_owned())?;
        let compact = compact
            .map(|thread| {
                thread
                    .join()
                    .map_err(|_| "compact wordcode artifact decoder panicked".to_owned())
            })
            .transpose()?;
        eprintln!(
            "compile-cache-phase: frontend-decode elapsed_ms={}",
            frontend_elapsed.as_millis()
        );
        eprintln!(
            "compile-cache-phase: executable-decode elapsed_ms={}",
            executable_elapsed.as_millis()
        );
        eprintln!(
            "compile-cache-phase: runtime-structural-decode elapsed_ms={}",
            structural_elapsed.as_millis()
        );
        if let Some((_, elapsed)) = &compact {
            eprintln!(
                "compile-cache-phase: compact-wordcode-decode elapsed_ms={}",
                elapsed.as_millis()
            );
        }
        Ok::<_, String>((
            compilation?,
            executable?,
            structural_seed?,
            compact
                .map(|(image, _)| image.map_err(|error| error.to_string()))
                .transpose()?,
        ))
    })?;
    eprintln!(
        "compile-cache-phase: parallel-section-decode elapsed_ms={}",
        parallel_phase.elapsed().as_millis()
    );
    if compilation.project().content_fingerprint().as_bytes() != &project_fingerprint {
        return Err(
            "compiled executable frontend fingerprint disagrees with its envelope".to_owned(),
        );
    }
    decode_and_attach_procedure_semantics(
        runtime_artifact
            .section(PROCEDURE_SEMANTICS_ARTIFACT_SECTION)
            .ok_or("runtime artifact is missing procedure semantic directory")?
            .payload(),
        executable.module_mut(),
    )?;
    if executable.module().deferred_procedure_count() != 0 {
        return Err("compiled executable contains deferred procedures".to_owned());
    }
    if let Some(stable_ids) = &stable_ids {
        validate_stable_linkage(&compilation, &executable, &structural_seed, stable_ids)?;
    }
    if let Some(compact) = compact {
        executable
            .module_mut()
            .attach_compact_wordcode(compact)
            .map_err(|error| format!("compact wordcode mismatch: {error}"))?;
    }
    let linked_runtime = runtime_artifact
        .section(RUNTIME_LINKED_ARTIFACT_SECTION)
        .map(|section| RuntimeImage::decode_linked_artifact(section.payload(), executable.module()))
        .transpose()
        .map_err(|error| format!("linked runtime artifact: {error}"))?;
    let lifecycle_index = runtime_artifact
        .section(LIFECYCLE_DIRECTORY_ARTIFACT_SECTION)
        .map(|section| LifecycleIndex::decode_portable(section.payload()))
        .transpose()
        .map_err(|error| format!("lifecycle directory artifact: {error}"))?;
    let default_map = runtime_artifact
        .section(DEFAULT_MAP_ARTIFACT_SECTION)
        .map(|section| dm_world::decode_named_portable_plan(section.payload()))
        .transpose()
        .map_err(|error| format!("default map artifact: {error}"))?;
    Ok((
        compilation,
        executable,
        structural_seed,
        linked_runtime,
        lifecycle_index,
        default_map,
    ))
}

fn validate_stable_linkage(
    compilation: &Compilation,
    executable: &ExecutableProcedures,
    structural_seed: &RuntimeStructuralSeed,
    stable_ids: &[dm_compiler::persistent_database::StableIdEntry],
) -> Result<(), String> {
    let expected_types = compilation
        .code_tree()
        .nodes()
        .iter()
        .filter(|node| node.kind == dm_object_tree::NodeKind::Type)
        .count();
    let mut types = stable_ids
        .iter()
        .filter(|entry| entry.namespace == "type")
        .collect::<Vec<_>>();
    if types.len() != expected_types {
        return Err("stable-ID type inventory disagrees with the frontend".to_owned());
    }
    types.sort_by_key(|entry| entry.id);
    for (runtime_id, entry) in types.into_iter().enumerate() {
        let path = TypePath::parse(&entry.name).map_err(|error| error.to_string())?;
        if structural_seed.type_id(&path) != u32::try_from(runtime_id).ok() {
            return Err(format!(
                "stable-ID type ordering disagrees for {}",
                entry.name
            ));
        }
    }

    let expected_procedures = compilation
        .code_tree()
        .nodes()
        .iter()
        .filter(|node| {
            matches!(
                node.kind,
                dm_object_tree::NodeKind::Procedure | dm_object_tree::NodeKind::Verb
            )
        })
        .count();
    let mut procedures = stable_ids
        .iter()
        .filter(|entry| entry.namespace == "procedure")
        .collect::<Vec<_>>();
    if procedures.len() != expected_procedures {
        return Err("stable-ID procedure inventory disagrees with the frontend".to_owned());
    }
    procedures.sort_by_key(|entry| entry.id);
    let mut previous = None;
    for entry in procedures {
        let id = executable
            .module()
            .effective_procedure_id(&entry.name)
            .ok_or_else(|| {
                format!(
                    "stable-ID procedure {} is absent from the module",
                    entry.name
                )
            })?;
        if previous.is_some_and(|previous| previous >= id) {
            return Err("stable-ID procedure ordering disagrees with the module".to_owned());
        }
        previous = Some(id);
    }
    Ok(())
}

fn executable_artifact_file(environment: &Path) -> PathBuf {
    environment.with_extension("d64")
}

fn project_cache_file(environment: &Path) -> PathBuf {
    let canonical = fs::canonicalize(environment).unwrap_or_else(|_| environment.to_path_buf());
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let cache_root = env::var_os("DREAM64_CACHE_DIR").map_or_else(
        || {
            env::current_exe()
                .ok()
                .and_then(|path| path.parent()?.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("target"))
                .join("dream64-cache")
        },
        PathBuf::from,
    );
    cache_root.join(format!("project-{hash:016x}.bin"))
}

fn compiler_database_file(project_cache: &Path) -> PathBuf {
    let name = project_cache
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("project");
    project_cache.with_file_name(format!("compiler-{name}.d64cdb"))
}

fn stable_ids_for_namespace(
    database: &PersistentCompilerDatabase,
    namespace: &str,
) -> BTreeMap<String, u64> {
    database
        .stable_ids
        .iter()
        .filter(|entry| entry.namespace == namespace)
        .map(|entry| (entry.name.clone(), entry.id))
        .collect()
}

fn validate_procedure_id_inventory(
    compilation: &Compilation,
    procedure_ids: &BTreeMap<String, u64>,
) -> Result<(), String> {
    let expected = compilation
        .code_tree()
        .nodes()
        .iter()
        .filter(|node| {
            matches!(
                node.kind,
                dm_object_tree::NodeKind::Procedure | dm_object_tree::NodeKind::Verb
            )
        })
        .count();
    if procedure_ids.len() != expected {
        return Err(format!(
            "persistent procedure ID inventory contains {} entries but the frontend contains {expected}",
            procedure_ids.len()
        ));
    }
    Ok(())
}

fn persistent_semantic_digest() -> [u8; 32] {
    digest32(
        b"dream64-compiler-semantics-v2",
        &engine_semantics_fingerprint(),
    )
}

fn persistent_build_configuration_digest() -> [u8; 32] {
    let compact = if env::var_os("DREAM64_DISABLE_COMPACT_WORDCODE").is_some() {
        b"compact=disabled".as_slice()
    } else {
        b"compact=enabled".as_slice()
    };
    digest32(b"dream64-compiler-build-v1", compact)
}

fn digest32(domain: &[u8], value: &[u8]) -> [u8; 32] {
    let mut left = md5::Context::new();
    left.consume(domain);
    left.consume([0]);
    left.consume(value);
    let mut right = md5::Context::new();
    right.consume(domain);
    right.consume([1]);
    right.consume(value);
    let mut digest = [0; 32];
    digest[..16].copy_from_slice(&left.compute().0);
    digest[16..].copy_from_slice(&right.compute().0);
    digest
}

fn ready_world_cache_file(
    project_cache: &Path,
    map_source: &str,
    compilation: &Compilation,
    production_identity: Option<&ProductionReadyWorldIdentity>,
) -> PathBuf {
    let mut identity = md5::Context::new();
    identity.consume(b"dream64-ready-world-v1");
    identity.consume(engine_semantics_fingerprint());
    identity.consume(env!("DREAM64_ENGINE_TARGET").as_bytes());
    identity.consume(compilation.project().content_fingerprint().as_bytes());
    identity.consume(map_source.as_bytes());
    if let Some(production_identity) = production_identity {
        identity.consume(b"production-ready-world-v1");
        identity.consume(production_identity.random_seed.to_le_bytes());
        identity.consume(production_identity.deployment_id.as_bytes());
    }
    let digest = identity.compute();
    project_cache.with_file_name(format!("ready-{digest:x}.bin"))
}

fn cached_world_plan(
    project_cache: &Path,
    map_source: &str,
    compilation: &Compilation,
) -> Result<dm_world::WorldPlan, String> {
    let path = project_cache.with_extension("mapplan");
    let project_fingerprint = *compilation.project().content_fingerprint().as_bytes();
    let (plan, stats) = dm_world::load_or_build_cached_plan(
        &path,
        map_source.as_bytes(),
        project_fingerprint,
        engine_semantics_fingerprint(),
        compilation,
    )
    .map_err(|error| error.to_string())?;
    eprintln!(
        "boot-progress: map-plan-cache cache={} artifact={} lookup_ms={} build_ms={} bytes={}",
        if stats.hit { "hit" } else { "miss" },
        path.display(),
        stats.lookup_elapsed.as_millis(),
        stats.build_elapsed.as_millis(),
        stats.written_bytes,
    );
    Ok(plan)
}

fn format_runtime_diagnostic(diagnostic: &RuntimeInitializerDiagnostic) -> String {
    format!(
        "runtime-initializer-diagnostic path={} storage={:?} category={:?} phase={:?} ordinal={} source={}:{}..{} blocker={}..{} message={:?}",
        diagnostic.variable_path,
        diagnostic.storage,
        diagnostic.category,
        diagnostic.phase,
        diagnostic.ordinal,
        diagnostic.source_path,
        diagnostic.initializer_span.start,
        diagnostic.initializer_span.end,
        diagnostic.blocker_span.start,
        diagnostic.blocker_span.end,
        diagnostic.message,
    )
}

fn master_controller_readiness(
    compilation: &Compilation,
    runtime: &RuntimeImage,
) -> Option<HeadlessReadinessProbe> {
    let has_master_type = runtime
        .types()
        .any(|(path, _)| path.as_str() == "/datum/controller/master");
    let expected = compilation
        .project()
        .object_macro("INITSTAGE_MAX")?
        .trim()
        .parse::<f32>()
        .ok()?;
    has_master_type.then(|| HeadlessReadinessProbe {
        qualified_storage: runtime
            .variables()
            .iter()
            .find(|variable| {
                variable.path.ends_with("/init_stage_completed")
                    && variable.path.contains("/datum/controller/master/")
            })
            .map(|variable| FieldName::static_storage(&variable.path)),
        global: FieldName::parse("Master").expect("DM global identifier is valid"),
        fields: if runtime.variables().iter().any(|variable| {
            variable.path.ends_with("/init_stage_completed")
                && variable.path.contains("/datum/controller/master/")
        }) {
            vec![]
        } else {
            vec![FieldName::parse("init_stage_completed").expect("DM field identifier is valid")]
        },
        expected: Value::number(expected),
    })
}

fn lobby_pregame_readiness(
    compilation: &Compilation,
    runtime: &RuntimeImage,
) -> Option<HeadlessReadinessProbe> {
    let has_ticker_type = runtime
        .types()
        .any(|(path, _)| path.as_str() == "/datum/controller/subsystem/ticker");
    let expected = compilation
        .project()
        .object_macro("GAME_STATE_PREGAME")?
        .trim()
        .parse::<f32>()
        .ok()?;
    has_ticker_type.then(|| HeadlessReadinessProbe {
        qualified_storage: None,
        global: FieldName::parse("SSticker").expect("DM global identifier is valid"),
        fields: vec![FieldName::parse("current_state").expect("DM field identifier is valid")],
        expected: Value::number(expected),
    })
}

fn print_compatibility_sweep(sweep: &dm_lifecycle::LifecycleCompatibilitySweep) {
    println!("sweep_targets={}", sweep.targets);
    println!("sweep_compatible={}", sweep.compatible);
    println!("sweep_issue_groups={}", sweep.issues.len());
    for issue in &sweep.issues {
        println!(
            "sweep_issue category={:?} locations={} message={:?}",
            issue.category,
            issue.locations.len(),
            issue.message
        );
        for location in &issue.locations {
            println!(
                "sweep_location phase={:?} procedure={} source={}:{}",
                location.kind,
                location.procedure_path,
                location.source.path,
                location.source.span.start
            );
        }
    }
}

fn print_plan_summary(
    map_path: &str,
    index: &LifecycleIndex,
    procedures: &ProcedureRegistry,
    plan: &dm_lifecycle::InitializationPlan,
) {
    println!("map={map_path}");
    println!("types={}", index.types().len());
    println!("procedures={}", procedures.procedures().len());
    println!("map_atoms={}", plan.map_atoms.len());
    println!("events={}", plan.events.len());
    println!("diagnostics={}", plan.diagnostics.len());
    println!("global_steps={}", plan.globals.initializer_steps);
    println!("global_constants={}", plan.globals.constants_materialized);
    println!(
        "global_unsupported={}",
        plan.globals.unsupported_initializers
    );
    let type_counts = type_lifecycle_counts(index);
    let atom_counts = plan.map_lifecycle_counts(index);
    for kind in [
        LifecycleKind::Genesis,
        LifecycleKind::New,
        LifecycleKind::Initialize,
        LifecycleKind::LateInitialize,
        LifecycleKind::Destroy,
    ] {
        println!(
            "type_{kind:?}={}",
            type_counts.get(&kind).copied().unwrap_or(0)
        );
        println!(
            "atom_{kind:?}={}",
            atom_counts.get(&kind).copied().unwrap_or(0)
        );
    }
    let mut diagnostic_counts = BTreeMap::new();
    for diagnostic in &plan.diagnostics {
        *diagnostic_counts.entry(diagnostic.kind).or_insert(0usize) += 1;
    }
    for (kind, count) in diagnostic_counts {
        println!("diagnostic_{kind:?}={count}");
    }
}

fn print_boot_summary(
    allocation: &dm_world::WorldAllocation,
    execution: &dm_lifecycle::InitializationExecution,
) {
    let stats = allocation.stats();
    println!("allocation_cells={}", stats.cells);
    println!("allocation_datums={}", stats.datums_allocated);
    println!("allocation_areas={}", stats.unique_areas);
    println!("allocation_turfs={}", stats.turfs);
    println!("allocation_movables={}", stats.movables);
    println!("allocation_deferred={}", allocation.work_items().len());
    println!("lifecycle_executed={}", execution.executed_events);
    println!("lifecycle_duplicates={}", execution.duplicate_map_events);
    println!("world_allocated={}", usize::from(execution.world.is_some()));
    println!("scheduler_tick={}", execution.scheduler.final_tick);
    println!("scheduler_rounds={}", execution.scheduler.rounds);
    println!(
        "scheduler_completed={}",
        execution.scheduler.completed_tasks
    );
    println!("scheduler_pending={}", execution.scheduler.pending_tasks);
    println!(
        "scheduler_termination={:?}",
        execution.scheduler.termination
    );
    for kind in [
        LifecycleKind::Genesis,
        LifecycleKind::New,
        LifecycleKind::Initialize,
        LifecycleKind::LateInitialize,
    ] {
        println!(
            "executed_{kind:?}={}",
            execution
                .executed_event_counts
                .get(&kind)
                .copied()
                .unwrap_or(0)
        );
    }
}

fn type_lifecycle_counts(index: &LifecycleIndex) -> BTreeMap<LifecycleKind, usize> {
    let mut counts = BTreeMap::new();
    for lifecycle in index.types() {
        for kind in [
            LifecycleKind::New,
            LifecycleKind::Initialize,
            LifecycleKind::LateInitialize,
            LifecycleKind::Destroy,
        ] {
            if matches!(
                lifecycle.targets.get(kind),
                LifecycleResolution::Resolved(_)
            ) {
                *counts.entry(kind).or_default() += 1;
            }
        }
    }
    counts
}

fn load_map(
    compilation: &Compilation,
    requested: Option<&Path>,
) -> Result<(String, String), String> {
    if let Some(path) = requested {
        let source =
            fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
        return Ok((path.display().to_string(), source));
    }
    let file = compilation
        .project()
        .files
        .iter()
        .find(|file| {
            file.relative_path
                .file_name()
                .is_some_and(|name| name.eq_ignore_ascii_case(OsStr::new("CentCom.dmm")))
        })
        .ok_or_else(|| {
            "compiled project contains no CentCom.dmm; pass its path explicitly".to_owned()
        })?;
    let source = file
        .text()
        .map_err(|error| format!("{}: {error}", file.relative_path.display()))?;
    Ok((file.relative_path.display().to_string(), source.to_owned()))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::persistent_database::PersistentCompilerDatabase;
    use dm_compiler::{BuildMode, CompilerDatabase};
    use dm_lifecycle::readiness_probe_matches;
    use dm_runtime::{RuntimeImage, RuntimeStructuralSeed};
    use dm_semantics::ProcedureRegistry;
    use dm_value::{FieldName, TypePath, Value};
    use dm_vm::{ExecutionState, execute_module_in_state};

    use super::{
        ArtifactSection, COMPACT_WORDCODE_ARTIFACT_SECTION, CompiledArtifact,
        ProductionReadyWorldIdentity, ReadyWorldMode, decode_compiled_executable,
        executable_artifact_file, fresh_launch_random_seed, launch_random_seed_from,
        lobby_pregame_readiness, parse_ready_world_mode, prepare_compiled_executable,
        ready_world_cache_file, restore_ready_world_cache, stable_ids_for_namespace,
        write_ready_world_cache,
    };

    static NEXT_SCRATCH: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        environment: PathBuf,
        source: PathBuf,
        cache: PathBuf,
        artifact: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let sequence = NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-executable-artifact-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&root).expect("scratch directory should be created");
            let environment = root.join("world.dme");
            let source = root.join("code.dm");
            let cache = root.join("project.bin");
            let artifact = root.join("project.d64");
            fs::write(
                &environment,
                "#include \"code.dm\"\n#include \"CentCom.dmm\"\n",
            )
            .expect("environment should be written");
            Self::write_source(&source, 1);
            fs::write(
                root.join("CentCom.dmm"),
                "\"a\" = (/turf, /area)\n(1,1,1) = {\"\na\n\"}\n",
            )
            .expect("map should be written");
            Self {
                root,
                environment,
                source,
                cache,
                artifact,
            }
        }

        fn write_source(path: &Path, value: u8) {
            fs::write(
                path,
                format!("/proc/make()\n\tvar/list/items = list({value})\n\treturn items\n"),
            )
            .expect("source should be written");
        }

        fn prepare(&self) -> super::PreparedCompiledExecutable {
            prepare_compiled_executable(&self.environment, &self.cache, &self.artifact, true)
                .expect("artifact preparation should succeed")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn runtime_artifact_is_a_lowercase_d64_sibling_of_the_environment() {
        let environment = Path::new("Monkestation2.0").join("tgstation.DmE");
        let runtime = executable_artifact_file(&environment);
        assert_eq!(runtime, Path::new("Monkestation2.0").join("tgstation.d64"));
    }

    #[test]
    fn lexically_earlier_additions_preserve_authoritative_runtime_ids() {
        let fixture = Fixture::new();
        let database_file = fixture.root.join("stable.d64cdb");
        let compiler = CompilerDatabase::new();
        let (first, _) = compiler
            .compile_persistent(
                &fixture.environment,
                &database_file,
                BuildMode::Incremental,
                [1; 32],
                [2; 32],
            )
            .unwrap();
        let first_database = PersistentCompilerDatabase::read(&database_file).unwrap();
        let first_procedure_ids = stable_ids_for_namespace(&first_database, "procedure");
        let first_type_ids = stable_ids_for_namespace(&first_database, "type");
        let first_registry =
            ProcedureRegistry::build_with_stable_ids(&first, &first_procedure_ids).unwrap();
        let first_seed = RuntimeStructuralSeed::build_with_stable_ids(&first, &first_type_ids)
            .expect("first structural link");
        let first_make = first_registry
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == "/proc/make")
            .unwrap()
            .id
            .index();

        fs::write(
            &fixture.source,
            "/aaa\n/aaa/proc/first()\n\treturn 0\n/proc/make()\n\treturn list(1)\n",
        )
        .unwrap();
        let (second, _) = compiler
            .compile_persistent(
                &fixture.environment,
                &database_file,
                BuildMode::Incremental,
                [1; 32],
                [2; 32],
            )
            .unwrap();
        let second_database = PersistentCompilerDatabase::read(&database_file).unwrap();
        let second_procedure_ids = stable_ids_for_namespace(&second_database, "procedure");
        let second_type_ids = stable_ids_for_namespace(&second_database, "type");
        let second_registry =
            ProcedureRegistry::build_with_stable_ids(&second, &second_procedure_ids).unwrap();
        let second_seed = RuntimeStructuralSeed::build_with_stable_ids(&second, &second_type_ids)
            .expect("second structural link");
        assert_eq!(
            second_registry
                .procedures()
                .iter()
                .find(|procedure| procedure.path.to_string() == "/proc/make")
                .unwrap()
                .id
                .index(),
            first_make
        );
        for path in first_type_ids.keys() {
            let path = TypePath::parse(path).unwrap();
            assert_eq!(first_seed.type_id(&path), second_seed.type_id(&path));
        }
    }

    #[test]
    fn nonsemantic_file_diff_reuses_persisted_executable_link() {
        let fixture = Fixture::new();
        let cold = fixture.prepare();
        assert!(cold.new_lowerings > 0);
        drop(cold);
        fs::remove_file(&fixture.artifact).unwrap();
        fs::write(
            &fixture.source,
            "/proc/make()\n\tvar/list/items = list(1)\n\treturn items\n// cache-neutral edit\n",
        )
        .unwrap();
        let warm = fixture.prepare();
        assert_eq!(warm.new_lowerings, 0);
        assert!(!warm.artifact_hit);
        assert_eq!(warm.executable.module().deferred_procedure_count(), 0);
        drop(warm);
        fs::remove_file(&fixture.artifact).unwrap();
        Fixture::write_source(&fixture.source, 2);
        let changed_body = fixture.prepare();
        assert!(changed_body.new_lowerings > 0);
    }

    #[test]
    fn launch_entropy_never_uses_the_deterministic_test_seed() {
        let first = fresh_launch_random_seed();
        let second = fresh_launch_random_seed();
        assert_ne!(first, 0);
        assert_ne!(second, 0);
        assert_ne!(first, second);
    }

    #[test]
    fn launch_seed_can_be_replayed_from_the_environment() {
        assert_eq!(
            launch_random_seed_from(Some("8675309")),
            (8_675_309, "environment")
        );
    }

    #[test]
    fn production_ready_world_modes_require_exclusive_complete_identity() {
        assert!(parse_ready_world_mode(true, true, false, false, Some("7"), Some("blue")).is_err());
        assert!(parse_ready_world_mode(true, false, false, false, None, Some("blue")).is_err());
        assert!(
            parse_ready_world_mode(true, false, false, false, Some("0"), Some("blue")).is_err()
        );
        assert!(parse_ready_world_mode(true, false, false, false, Some("7"), Some(" ")).is_err());
        assert_eq!(
            parse_ready_world_mode(true, false, true, false, Some("7"), Some(" blue ")).unwrap(),
            ReadyWorldMode::Prewarm(ProductionReadyWorldIdentity {
                random_seed: 7,
                deployment_id: "blue".to_owned(),
            })
        );
        assert_eq!(
            parse_ready_world_mode(false, true, false, false, Some("9"), Some("green")).unwrap(),
            ReadyWorldMode::Activate(ProductionReadyWorldIdentity {
                random_seed: 9,
                deployment_id: "green".to_owned(),
            })
        );
        assert_eq!(
            parse_ready_world_mode(true, true, true, true, None, None).unwrap(),
            ReadyWorldMode::Disabled,
            "the explicit disable switch overrides every cache mode"
        );
        assert_eq!(
            parse_ready_world_mode(false, false, true, false, None, None).unwrap(),
            ReadyWorldMode::Development
        );
    }

    #[test]
    fn monk_lobby_probe_requires_ticker_pregame_state() {
        let fixture = Fixture::new();
        fs::write(
            &fixture.source,
            concat!(
                "#define GAME_STATE_PREGAME 1\n",
                "var/global/datum/controller/subsystem/ticker/SSticker\n",
                "/datum/controller/subsystem/ticker\n",
                "\tvar/current_state = 0\n",
            ),
        )
        .unwrap();
        let prepared = fixture.prepare();
        let runtime = RuntimeImage::from_compilation(&prepared.compilation).unwrap();
        let probe = lobby_pregame_readiness(&prepared.compilation, &runtime)
            .expect("Monke-like ticker should expose a pregame probe");

        let mut state = ExecutionState::new();
        let ticker = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/controller/subsystem/ticker").unwrap());
        state
            .heap_mut()
            .set_datum_field(
                ticker,
                FieldName::parse("current_state").unwrap(),
                Value::number(0.0),
            )
            .unwrap();
        state.set_global(FieldName::parse("SSticker").unwrap(), Value::Datum(ticker));
        assert!(!readiness_probe_matches(&state, &probe));

        state
            .heap_mut()
            .set_datum_field(
                ticker,
                FieldName::parse("current_state").unwrap(),
                Value::number(1.0),
            )
            .unwrap();
        assert!(readiness_probe_matches(&state, &probe));
    }

    #[test]
    fn ready_world_identity_changes_with_map_content_and_compressed_state_roundtrips() {
        let fixture = Fixture::new();
        let prepared = fixture.prepare();
        let first = ready_world_cache_file(
            &fixture.cache,
            "(1,1,1) = {\"a\"}",
            &prepared.compilation,
            None,
        );
        let second = ready_world_cache_file(
            &fixture.cache,
            "(1,1,1) = {\"b\"}",
            &prepared.compilation,
            None,
        );
        assert_ne!(first, second);
        let deployment = |random_seed, deployment_id: &str| ProductionReadyWorldIdentity {
            random_seed,
            deployment_id: deployment_id.to_owned(),
        };
        let production = ready_world_cache_file(
            &fixture.cache,
            "(1,1,1) = {\"a\"}",
            &prepared.compilation,
            Some(&deployment(41, "blue")),
        );
        assert_ne!(production, first);
        assert_ne!(
            production,
            ready_world_cache_file(
                &fixture.cache,
                "(1,1,1) = {\"a\"}",
                &prepared.compilation,
                Some(&deployment(42, "blue")),
            )
        );
        assert_ne!(
            production,
            ready_world_cache_file(
                &fixture.cache,
                "(1,1,1) = {\"a\"}",
                &prepared.compilation,
                Some(&deployment(41, "green")),
            )
        );

        let mut state = ExecutionState::new();
        state.set_global(FieldName::parse("answer").unwrap(), Value::number(42.0));
        let bytes = write_ready_world_cache(&first, &state).unwrap();
        assert!(bytes > 0);
        let mut restored = ExecutionState::new();
        assert_eq!(
            restore_ready_world_cache(&first, &mut restored, prepared.executable.module()).unwrap(),
            bytes
        );
        assert_eq!(
            restored.global(&FieldName::parse("answer").unwrap()),
            Some(&Value::number(42.0))
        );
    }

    #[test]
    fn unchanged_artifact_hits_and_source_byte_change_rebuilds() {
        let fixture = Fixture::new();
        let first = fixture.prepare();
        assert!(!first.artifact_hit);
        assert!(first.new_lowerings > 0);
        let first_fingerprint = first.compilation.project().content_fingerprint();
        let artifact_bytes = fs::read(&fixture.artifact).expect("artifact should exist");
        drop(first);

        let warm = fixture.prepare();
        assert!(warm.artifact_hit);
        assert_eq!(warm.new_lowerings, 0);
        assert!(warm.project_snapshot_hit);
        assert_eq!(
            fs::read(&fixture.artifact).expect("artifact should remain readable"),
            artifact_bytes,
            "a hit must not rewrite or recompile the executable"
        );
        drop(warm);

        let original_metadata =
            fs::metadata(&fixture.source).expect("source metadata should exist");
        Fixture::write_source(&fixture.source, 2);
        let changed_metadata =
            fs::metadata(&fixture.source).expect("changed metadata should exist");
        assert_eq!(changed_metadata.len(), original_metadata.len());
        assert_ne!(
            changed_metadata.modified().unwrap(),
            original_metadata.modified().unwrap()
        );
        let changed = fixture.prepare();
        assert!(!changed.artifact_hit);
        assert!(changed.new_lowerings > 0);
        assert_ne!(
            changed.compilation.project().content_fingerprint(),
            first_fingerprint
        );
    }

    #[test]
    fn runtime_mode_never_compiles_a_missing_artifact() {
        let fixture = Fixture::new();
        drop(fixture.prepare());
        let runtime = prepare_compiled_executable(
            &fixture.environment,
            &fixture.cache,
            &fixture.artifact,
            false,
        )
        .expect("runtime should load a compiler-produced artifact");
        assert!(runtime.artifact_hit);
        drop(runtime);

        fs::remove_file(&fixture.artifact).expect("runtime artifact should be removed");
        let error = prepare_compiled_executable(
            &fixture.environment,
            &fixture.cache,
            &fixture.artifact,
            false,
        )
        .err()
        .expect("runtime must fail instead of compiling");
        assert!(error.contains("run `dream64-compiler"));
        assert!(!fixture.artifact.exists());
    }

    #[test]
    fn standalone_runtime_loads_only_the_compiled_d64() {
        let fixture = Fixture::new();
        let compiled = fixture.prepare();
        let expected = compiled.compilation.project().content_fingerprint();
        drop(compiled);
        let artifact =
            CompiledArtifact::read_from(&fixture.artifact, *expected.as_bytes()).unwrap();
        let sections = artifact
            .sections()
            .iter()
            .filter(|section| section.id() != super::COMPILATION_ARTIFACT_SECTION)
            .cloned()
            .collect();
        CompiledArtifact::new(*expected.as_bytes(), sections)
            .unwrap()
            .write_atomic(&fixture.artifact)
            .unwrap();
        fs::remove_file(&fixture.environment).expect("source environment should be removable");
        fs::remove_file(&fixture.source).expect("source file should be removable");
        fs::remove_file(fixture.root.join("CentCom.dmm")).expect("map source should be removable");

        let runtime = super::prepare_standalone_runtime(&fixture.artifact)
            .expect("self-contained runtime should not require compiler sources");
        let artifact = CompiledArtifact::read_from(&fixture.artifact, *expected.as_bytes())
            .expect("runtime artifact should decode");
        assert!(
            artifact
                .section(super::COMPILATION_ARTIFACT_SECTION)
                .is_none()
        );
        assert!(
            artifact
                .section(super::RUNTIME_LINKED_ARTIFACT_SECTION)
                .is_some()
        );
        assert!(!runtime.map_path.is_empty());
        assert!(!runtime.world.cells().is_empty());
        assert!(runtime.lifecycle.find_path("/world").is_some());
        assert_eq!(
            runtime
                .dmm_measurements
                .get("centcom.dmm")
                .expect("source-deleted artifact retains DMM measurement")
                .measurement
                .bounds,
            [1, 1, 1, 1, 1, 1]
        );
        drop(runtime);
        assert_eq!(
            super::run_standalone_linked_boot(
                &fixture.artifact,
                true,
                None,
                std::time::Instant::now(),
            ),
            std::process::ExitCode::SUCCESS,
        );
    }

    #[test]
    fn source_deleted_standalone_ticker_uses_portable_readiness_manifest() {
        let fixture = Fixture::new();
        fs::write(
            &fixture.source,
            concat!(
                "#define GAME_STATE_PREGAME 2\n",
                "/datum/controller/subsystem/ticker\n",
                "\tvar/current_state = GAME_STATE_PREGAME\n",
                "/var/global/datum/controller/subsystem/ticker/SSticker = new\n",
            ),
        )
        .unwrap();
        let compiled = fixture.prepare();
        let fingerprint = *compiled
            .compilation
            .project()
            .content_fingerprint()
            .as_bytes();
        drop(compiled);
        let artifact = CompiledArtifact::read_from(&fixture.artifact, fingerprint).unwrap();
        let sections = artifact
            .sections()
            .iter()
            .filter(|section| section.id() != super::COMPILATION_ARTIFACT_SECTION)
            .cloned()
            .collect();
        CompiledArtifact::new(fingerprint, sections)
            .unwrap()
            .write_atomic(&fixture.artifact)
            .unwrap();
        fs::remove_file(&fixture.environment).unwrap();
        fs::remove_file(&fixture.source).unwrap();

        let prepared = super::prepare_standalone_runtime(&fixture.artifact).unwrap();
        let probe = prepared
            .readiness
            .as_ref()
            .expect("ticker artifact requires readiness");
        assert_eq!(probe.expected, Value::number(2.0));
        assert_eq!(probe.global.as_str(), "SSticker");
        assert_eq!(probe.fields[0].as_str(), "current_state");

        let artifact = CompiledArtifact::read_from(&fixture.artifact, fingerprint).unwrap();
        let sections = artifact
            .sections()
            .iter()
            .filter(|section| section.id() != super::BOOT_MANIFEST_ARTIFACT_SECTION)
            .cloned()
            .collect();
        let missing = fixture.root.join("missing-manifest.d64");
        CompiledArtifact::new(fingerprint, sections)
            .unwrap()
            .write_atomic(&missing)
            .unwrap();
        assert!(
            super::prepare_standalone_runtime(&missing)
                .err()
                .unwrap()
                .contains("missing its boot readiness manifest")
        );
    }

    #[test]
    fn engine_mismatch_and_corruption_each_fall_back_once() {
        let fixture = Fixture::new();
        drop(fixture.prepare());

        let mut wrong_engine = fs::read(&fixture.artifact).expect("artifact should exist");
        wrong_engine[56] ^= 0xff;
        refresh_envelope_checksums(&mut wrong_engine);
        fs::write(&fixture.artifact, wrong_engine).expect("mismatched artifact should be written");
        let rebuilt_engine = fixture.prepare();
        assert!(!rebuilt_engine.artifact_hit);
        assert!(
            rebuilt_engine
                .miss_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("engine semantics mismatch"))
        );
        drop(rebuilt_engine);

        let mut corrupt = fs::read(&fixture.artifact).expect("rebuilt artifact should exist");
        let payload_offset = corrupt.len() / 2;
        corrupt[payload_offset] ^= 0xff;
        fs::write(&fixture.artifact, corrupt).expect("corrupt artifact should be written");
        let rebuilt_corrupt = fixture.prepare();
        assert!(!rebuilt_corrupt.artifact_hit);
        assert!(
            rebuilt_corrupt
                .miss_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("checksum mismatch"))
        );
    }

    #[test]
    fn independently_loaded_executables_use_distinct_heaps() {
        let fixture = Fixture::new();
        let prepared = fixture.prepare();
        let fingerprint = *prepared
            .compilation
            .project()
            .content_fingerprint()
            .as_bytes();
        let semantic_entry = prepared
            .procedures
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == "/proc/make")
            .and_then(|procedure| procedure.effective_target)
            .expect("fixture semantic entry should exist");
        drop(prepared);
        let (_, left, _, _, _, _) = decode_compiled_executable(&fixture.artifact, fingerprint)
            .expect("first executable should load");
        let (_, right, _, _, _, _) = decode_compiled_executable(&fixture.artifact, fingerprint)
            .expect("second executable should load");
        let left_entry = left
            .implementation(semantic_entry)
            .expect("first fixture entry should exist");
        let right_entry = right
            .implementation(semantic_entry)
            .expect("second fixture entry should exist");
        let mut left_state = ExecutionState::new();
        let mut right_state = ExecutionState::new();
        let Value::List(left_list) =
            execute_module_in_state(left.module(), left_entry, &[], &mut left_state)
                .expect("first execution should succeed")
        else {
            panic!("first execution should return a list")
        };
        let Value::List(right_list) =
            execute_module_in_state(right.module(), right_entry, &[], &mut right_state)
                .expect("second execution should succeed")
        else {
            panic!("second execution should return a list")
        };
        left_state
            .heap_mut()
            .list_mut(left_list)
            .expect("first result should be live")
            .add(Value::number(99.0));
        assert_eq!(left_state.heap().list(left_list).unwrap().len(), 2);
        assert_eq!(right_state.heap().list(right_list).unwrap().len(), 1);
        assert_eq!(left_state.heap().live_list_count(), 1);
        assert_eq!(right_state.heap().live_list_count(), 1);
    }

    #[test]
    fn optional_compact_wordcode_section_loads_and_attaches() {
        let fixture = Fixture::new();
        let prepared = fixture.prepare();
        let fingerprint = *prepared
            .compilation
            .project()
            .content_fingerprint()
            .as_bytes();
        let compact = dm_vm::CompactWordcodeImage::build(prepared.executable.module())
            .expect("compact image should build")
            .encode()
            .expect("compact image should encode");
        drop(prepared);

        let artifact = CompiledArtifact::read_from(&fixture.artifact, fingerprint)
            .expect("base artifact should load");
        let mut sections: Vec<_> = artifact
            .sections()
            .iter()
            .filter(|section| section.id() != COMPACT_WORDCODE_ARTIFACT_SECTION)
            .cloned()
            .collect();
        let stable_ids = sections.pop().expect("stable-ID section should exist");
        sections.push(ArtifactSection::new(
            COMPACT_WORDCODE_ARTIFACT_SECTION,
            compact,
        ));
        sections.push(stable_ids);
        CompiledArtifact::new(fingerprint, sections)
            .expect("artifact with compact wordcode should build")
            .write_atomic(&fixture.artifact)
            .expect("artifact with compact wordcode should be written");

        let (_, executable, _, _, _, _) =
            decode_compiled_executable(&fixture.artifact, fingerprint)
                .expect("executable with compact wordcode should load");
        let compact = executable
            .module()
            .compact_wordcode()
            .expect("compact image should attach");
        assert_eq!(
            compact.procedure_count(),
            executable.module().procedure_count()
        );
        assert!(compact.word_count() > 0);
        assert!(compact.specialized_word_count() > 0);
    }

    fn refresh_envelope_checksums(bytes: &mut [u8]) {
        let header_length = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
        let header_checksum_offset = header_length - 4;
        let header_checksum = crc32fast::hash(&bytes[..header_checksum_offset]);
        bytes[header_checksum_offset..header_length]
            .copy_from_slice(&header_checksum.to_le_bytes());
        let footer = bytes.len() - 32;
        let body_checksum = crc32fast::hash(&bytes[..footer]);
        bytes[footer + 16..footer + 20].copy_from_slice(&body_checksum.to_le_bytes());
        let footer_checksum = crc32fast::hash(&bytes[footer..footer + 28]);
        bytes[footer + 28..footer + 32].copy_from_slice(&footer_checksum.to_le_bytes());
    }
}
