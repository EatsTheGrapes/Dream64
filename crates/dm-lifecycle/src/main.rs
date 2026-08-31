use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::CompilerDatabase;
use dm_lifecycle::ipc::{BootPhase, LoopbackIpc, parse_loopback_address};
use dm_lifecycle::{
    LifecycleIndex, SchedulerDrainTermination, audit_initialization_plan_with_precompiled,
    build_initialization_plan, execute_boot_initialization_plan_with_precompiled,
    execute_boot_initialization_plan_with_precompiled_and_startup_service,
    precompile_lifecycle_for_world, precompile_lifecycle_for_world_with_executable,
    sweep_lifecycle_compatibility, sweep_lifecycle_compatibility_with_closures,
};
use dm_runtime::{RuntimeImage, RuntimeImageConstructionEvent};
use dm_semantics::{ExecutableProcedures, ProcedureRegistry};
use dm_world::allocate_world;

mod server;

use server::artifact_pipeline::{
    PreparedCacheStats, cached_world_plan, executable_artifact_file, prepare_compiled_executable,
    prepare_standalone_artifact, project_cache_file, run_standalone_linked_boot,
};
use server::cli::{Command, ReadyWorldMode, progress_label, ready_world_mode_from_environment};
use server::lobby_preflight::run_lobby_preflight;
use server::ready_world::{
    ready_world_cache_file, restore_ready_world_cache, write_ready_world_cache,
};
use server::reporting::{
    format_runtime_diagnostic, inspect_ready_globals, load_map, lobby_pregame_readiness,
    master_controller_readiness, print_boot_summary, print_compatibility_sweep, print_plan_summary,
};
use server::server_loop::{
    fresh_launch_random_seed, launch_random_seed, report_public_endpoint,
    run_persistent_server_loop, run_prewarmed_standby, startup_scheduler_limits,
};

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
