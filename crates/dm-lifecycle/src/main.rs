use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use dm_compiler::{Compilation, CompilerDatabase};
use dm_lifecycle::ipc::{LoopbackIpc, parse_loopback_address};
use dm_lifecycle::{
    HeadlessReadinessProbe, HostSliceBudget, LifecycleIndex, LifecycleKind, LifecycleResolution,
    SchedulerDrainLimits, SchedulerDrainTermination, advance_persistent_scheduler_responsive,
    artifact::{ArtifactSection, CompiledArtifact, engine_semantics_fingerprint},
    audit_initialization_plan_with_precompiled, build_initialization_plan,
    execute_boot_initialization_plan_with_precompiled,
    execute_boot_initialization_plan_with_precompiled_and_startup_service,
    precompile_lifecycle_for_world, precompile_lifecycle_for_world_with_executable,
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

#[allow(clippy::too_many_lines)]
fn run_main() -> ExitCode {
    let process_started = Instant::now();
    let mut arguments = env::args_os().skip(1);
    let Some(first) = arguments.next() else {
        eprintln!(
            "usage: dm-lifecycle [compile|plan|boot|sweep|sweep-closure|lobby-preflight|lobby-preview] <world.dme> [map.dmm]"
        );
        return ExitCode::from(2);
    };
    let (command, environment) = if first.as_os_str() == OsStr::new("compile") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dm-lifecycle compile <world.dme>");
            return ExitCode::from(2);
        };
        (Command::Compile, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("plan") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dm-lifecycle plan <world.dme> [map.dmm]");
            return ExitCode::from(2);
        };
        (Command::Plan, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("boot") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dm-lifecycle boot <world.dme> [map.dmm]");
            return ExitCode::from(2);
        };
        (Command::Boot, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("sweep") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dm-lifecycle sweep <world.dme> [map.dmm]");
            return ExitCode::from(2);
        };
        (Command::Sweep, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("sweep-closure") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dm-lifecycle sweep-closure <world.dme> [map.dmm]");
            return ExitCode::from(2);
        };
        (Command::SweepClosure, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("lobby-preflight") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dm-lifecycle lobby-preflight <world.dme>");
            return ExitCode::from(2);
        };
        (Command::LobbyPreflight, PathBuf::from(environment))
    } else if first.as_os_str() == OsStr::new("lobby-preview") {
        let Some(environment) = arguments.next() else {
            eprintln!("usage: dm-lifecycle lobby-preview <world.dme> <world-params>");
            return ExitCode::from(2);
        };
        (Command::LobbyPreview, PathBuf::from(environment))
    } else {
        (Command::Plan, PathBuf::from(first))
    };
    let requested_map = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        eprintln!(
            "usage: dm-lifecycle [compile|plan|boot|sweep|sweep-closure|lobby-preflight] <world.dme> [map.dmm]"
        );
        return ExitCode::from(2);
    }
    if command == Command::Compile && requested_map.is_some() {
        eprintln!("usage: dm-lifecycle {} <world.dme>", "compile");
        return ExitCode::from(2);
    }
    let audit_runtime =
        command == Command::Boot && env::var_os("DREAM64_BOOT_AUDIT_RUNTIME").is_some();
    let mut boot_startup_ipc = if command == Command::Boot && !audit_runtime {
        let ipc_address =
            env::var("DREAM64_IPC_ADDR").unwrap_or_else(|_| "127.0.0.1:51664".to_owned());
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
    let cache_file = project_cache_file(&environment);
    let artifact_file = executable_artifact_file(&environment);
    let mut cached_executable = None;
    let mut cached_procedures = None;
    let mut cached_structural_seed = None;
    let compilation_result: Result<_, String> = if cached_compilation {
        prepare_compiled_executable(&environment, &cache_file, &artifact_file).map(|prepared| {
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
            "{progress}: project-compile-complete elapsed_ms={} preprocessing_cache={} parsed_syntax_cache={} cache={}",
            compile_started.elapsed().as_millis(),
            if project_cache.is_some_and(|cache| cache.project_snapshot_hit) {
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
            ipc.set_startup_phase("Loading and parsing the map");
        }
        eprintln!("boot-progress: loading map");
        let (map_path, map_source) = match load_map(&compilation, requested_map.as_deref()) {
            Ok(map) => map,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
        eprintln!("boot-progress: preparing map plan {map_path}");
        let world = match cached_world_plan(&cache_file, &map_source, &compilation) {
            Ok(world) => world,
            Err(error) => {
                eprintln!("{map_path}: {error}");
                return ExitCode::FAILURE;
            }
        };
        drop(map_source);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        if let Some(ipc) = &boot_startup_ipc {
            ipc.set_startup_phase("Precompiling lifecycle procedures");
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
            None => {
                precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
            }
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
        prepared_boot = Some((map_path, world, precompiled));
    }
    if command == Command::Boot {
        if let Some(ipc) = &boot_startup_ipc {
            ipc.set_startup_phase("Materializing globals and type defaults");
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
    let runtime_result = if let Some((_, _, precompiled)) = prepared_boot.as_mut() {
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
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
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
            ipc.set_startup_phase("Indexing lifecycle dispatch");
        }
        eprintln!("boot-progress: indexing procedures and lifecycle dispatch");
    }
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let mut boot_precompiled = None;
    let (map_path, world) = if let Some((map_path, world, precompiled)) = prepared_boot.take() {
        boot_precompiled = Some(precompiled);
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
        if let Some(ipc) = &startup_ipc {
            ipc.set_startup_phase("Preflighting map initializer plans");
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
            ipc.set_startup_phase("Allocating the map world");
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
            ipc.set_startup_phase("Starting world and subsystem controller");
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
            let mut startup_lobby_visible = false;
            let mut service_startup_clients =
                |executable: &ExecutableProcedures, state: &mut dm_vm::ExecutionState| {
                    if !startup_lobby_visible {
                        ipc.set_startup_phase("Initializing subsystems");
                        ipc.show_startup_lobby();
                        startup_lobby_visible = true;
                        eprintln!("server-progress: startup-lobby=read-only");
                    }
                    ipc.apply_executable_tick_boundary(executable, state);
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
        eprintln!(
            "boot-progress: headless ready; entering persistent scheduler loop ready_elapsed_ms={}",
            process_started.elapsed().as_millis()
        );
        if let Some(ipc) = &startup_ipc {
            ipc.set_startup_phase("Server ready");
            ipc.accept_startup_clients();
            if let Some(state) = precompiled.persistent_state_mut() {
                ipc.enable_session_interaction(state);
            }
            eprintln!(
                "server-progress: loopback-ipc={} startup=accepting",
                ipc.local_addr()
            );
        }
        let mut ipc_address = startup_ipc.expect("production boot bound loopback IPC");
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
        let mut host_budget =
            HostSliceBudget::new(100_000, 1_000, 100_000, Duration::from_millis(10));
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
                &mut runtime,
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
            over_target_slices = over_target_slices
                .saturating_add(u64::from(vm_elapsed > Duration::from_millis(10)));
            host_budget.observe(vm_elapsed);
            // A VM slice may have consumed the complete latency allowance.
            // Pump transport again before logging or sleeping so ping/input and
            // completed immutable worker jobs never wait for the next tick.
            ipc_address.apply_lifecycle_tick_boundary(precompiled);
            slices = slices.saturating_add(1);
            if slices == 1 || slices % 100 == 0 {
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
    ExitCode::SUCCESS
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
            env::var("DREAM64_IPC_ADDR").unwrap_or_else(|_| "127.0.0.1:51664".to_owned());
        let mut ipc = match parse_loopback_address(&ipc_address).and_then(LoopbackIpc::bind) {
            Ok(ipc) => ipc,
            Err(error) => {
                eprintln!("lobby-preview: loopback IPC: {error}");
                return ExitCode::FAILURE;
            }
        };
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
    project_snapshot_hit: bool,
    parsed_syntax_hit: Option<bool>,
    artifact_hit: bool,
    miss_reason: Option<String>,
    new_lowerings: usize,
}

fn prepare_compiled_executable(
    environment: &Path,
    cache_file: &Path,
    artifact_file: &Path,
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
            Ok((compilation, executable, structural_seed)) => {
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
    drop(project);

    // One miss performs exactly one ordinary frontend compilation followed by
    // one complete symbolic link and deterministic eager materialization.
    let phase = Instant::now();
    let (compilation, cache_stats) = CompilerDatabase::new()
        .compile_cached_with_stats(environment, cache_file)
        .map_err(|error| error.to_string())?;
    eprintln!(
        "compile-cache-phase: frontend-compile elapsed_ms={} parsed_syntax_hit={}",
        phase.elapsed().as_millis(),
        cache_stats.parsed_syntax_hit
    );
    let phase = Instant::now();
    let procedures = ProcedureRegistry::build(&compilation);
    eprintln!(
        "compile-cache-phase: procedure-registry elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let phase = Instant::now();
    let executable = procedures
        .compile_vm_all_symbolic_deferred(&compilation)
        .map_err(|error| format!("complete executable lowering: {error}"))?;
    eprintln!(
        "compile-cache-phase: symbolic-lowering elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let phase = Instant::now();
    let executable = executable
        .into_fully_eager_bounded(MAX_EAGER_ARTIFACT_DIAGNOSTICS)
        .map_err(|error| format!("complete executable lowering: {error}"))?;
    eprintln!(
        "compile-cache-phase: eager-materialization elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    if executable.module().deferred_procedure_count() != 0 {
        return Err("complete executable lowering retained deferred procedures".to_owned());
    }
    let new_lowerings = executable.module().procedure_count();
    let phase = Instant::now();
    let compilation_payload = compilation.encode_compiled_artifact();
    eprintln!(
        "compile-cache-phase: frontend-encode elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let phase = Instant::now();
    let executable_payload = executable.encode_compiled_artifact()?;
    eprintln!(
        "compile-cache-phase: executable-encode elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let phase = Instant::now();
    let structural_seed = RuntimeStructuralSeed::build(&compilation)
        .map_err(|error| format!("runtime structural seed: {error}"))?;
    let structural_payload = structural_seed.encode_compiled_artifact();
    eprintln!(
        "compile-cache-phase: runtime-structural-encode elapsed_ms={} bytes={}",
        phase.elapsed().as_millis(),
        structural_payload.len(),
    );
    eprintln!(
        "compile-progress: executable-artifact-payloads frontend_bytes={} executable_bytes={}",
        compilation_payload.len(),
        executable_payload.len(),
    );
    let fingerprint = *compilation.project().content_fingerprint().as_bytes();
    let phase = Instant::now();
    let artifact = CompiledArtifact::new(
        fingerprint,
        vec![
            ArtifactSection::new(COMPILATION_ARTIFACT_SECTION, compilation_payload),
            ArtifactSection::new(EXECUTABLE_ARTIFACT_SECTION, executable_payload),
            ArtifactSection::new(RUNTIME_STRUCTURAL_ARTIFACT_SECTION, structural_payload),
        ],
    )
    .map_err(|error| format!("build executable artifact: {error}"))?;
    eprintln!(
        "compile-cache-phase: envelope-build elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let phase = Instant::now();
    let write_stats = artifact
        .write_atomic_with_stats(artifact_file)
        .map_err(|error| format!("write executable artifact: {error}"))?;
    eprintln!(
        "compile-cache-phase: artifact-write elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    eprintln!(
        "compile-progress: executable-artifact-write encoded_bytes={} payload_bytes={} peak_staging_bytes={} write_calls={}",
        write_stats.encoded_bytes,
        write_stats.payload_bytes,
        write_stats.peak_staging_bytes,
        write_stats.write_calls,
    );
    Ok(PreparedCompiledExecutable {
        compilation,
        procedures,
        executable,
        structural_seed,
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
) -> Result<(Compilation, ExecutableProcedures, RuntimeStructuralSeed), String> {
    let phase = Instant::now();
    let artifact = match CompiledArtifact::read_from(artifact_file, project_fingerprint) {
        Ok(artifact) => artifact,
        Err(error) => {
            eprintln!(
                "compile-cache-phase: artifact-read-checksum-rejected elapsed_ms={} reason={error}",
                phase.elapsed().as_millis()
            );
            return Err(error.to_string());
        }
    };
    eprintln!(
        "compile-cache-phase: artifact-read-checksum elapsed_ms={}",
        phase.elapsed().as_millis()
    );
    let storage = artifact.storage_stats();
    eprintln!(
        "boot-progress: executable-artifact-storage payload_bytes={} backing_bytes={} backing_allocations={}",
        storage.payload_bytes, storage.backing_bytes, storage.backing_allocations,
    );
    if artifact.sections().len() != 3 {
        return Err(format!(
            "compiled executable contains {} sections instead of 3",
            artifact.sections().len()
        ));
    }
    let frontend_payload = artifact
        .section(COMPILATION_ARTIFACT_SECTION)
        .ok_or_else(|| "compiled executable is missing the frontend section".to_owned())?
        .payload();
    let executable_payload = artifact
        .section(EXECUTABLE_ARTIFACT_SECTION)
        .ok_or_else(|| "compiled executable is missing the bytecode section".to_owned())?
        .payload();
    let structural_payload = artifact
        .section(RUNTIME_STRUCTURAL_ARTIFACT_SECTION)
        .ok_or_else(|| "compiled executable is missing the runtime structural section".to_owned())?
        .payload();
    let parallel_phase = Instant::now();
    let (compilation, executable, structural_seed) = std::thread::scope(|scope| {
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
        let (compilation, frontend_elapsed) = frontend
            .join()
            .map_err(|_| "frontend artifact decoder panicked".to_owned())?;
        let (executable, executable_elapsed) = bytecode
            .join()
            .map_err(|_| "bytecode artifact decoder panicked".to_owned())?;
        let (structural_seed, structural_elapsed) = structural
            .join()
            .map_err(|_| "runtime structural artifact decoder panicked".to_owned())?;
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
        Ok::<_, String>((compilation?, executable?, structural_seed?))
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
    if executable.module().deferred_procedure_count() != 0 {
        return Err("compiled executable contains deferred procedures".to_owned());
    }
    Ok((compilation, executable, structural_seed))
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
    println!("lifecycle_executed={}", execution.events.len());
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
    let mut counts = BTreeMap::new();
    for event in &execution.events {
        let dm_lifecycle::InitializationEvent::Lifecycle { kind, .. } = event.event else {
            continue;
        };
        *counts.entry(kind).or_insert(0usize) += 1;
    }
    for kind in [
        LifecycleKind::Genesis,
        LifecycleKind::New,
        LifecycleKind::Initialize,
        LifecycleKind::LateInitialize,
    ] {
        println!(
            "executed_{kind:?}={}",
            counts.get(&kind).copied().unwrap_or(0)
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

    use dm_value::Value;
    use dm_vm::{ExecutionState, execute_module_in_state};

    use super::{
        decode_compiled_executable, executable_artifact_file, prepare_compiled_executable,
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
            fs::write(&environment, "#include \"code.dm\"\n")
                .expect("environment should be written");
            Self::write_source(&source, 1);
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
            prepare_compiled_executable(&self.environment, &self.cache, &self.artifact)
                .expect("artifact preparation should succeed")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn executable_artifact_is_a_lowercase_d64_sibling_of_the_environment() {
        let environment = Path::new("Monkestation2.0").join("tgstation.DmE");
        assert_eq!(
            executable_artifact_file(&environment),
            Path::new("Monkestation2.0").join("tgstation.d64")
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
        let (_, left, _) = decode_compiled_executable(&fixture.artifact, fingerprint)
            .expect("first executable should load");
        let (_, right, _) = decode_compiled_executable(&fixture.artifact, fingerprint)
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
