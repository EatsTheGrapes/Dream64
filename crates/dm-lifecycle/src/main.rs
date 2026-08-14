use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::{Compilation, CompilerDatabase};
use dm_lifecycle::{
    HeadlessReadinessProbe, LifecycleIndex, LifecycleKind, LifecycleResolution,
    SchedulerDrainLimits, SchedulerDrainTermination, advance_persistent_scheduler,
    artifact::{ArtifactSection, CompiledArtifact},
    audit_initialization_plan_with_precompiled, build_initialization_plan,
    execute_boot_initialization_plan_with_precompiled, precompile_lifecycle_for_world,
    precompile_lifecycle_for_world_with_executable, sweep_lifecycle_compatibility,
    sweep_lifecycle_compatibility_with_closures,
};
use dm_project::Project;
use dm_runtime::{RuntimeImage, RuntimeImageConstructionEvent, RuntimeInitializerDiagnostic};
use dm_semantics::{ExecutableProcedures, ProcedureRegistry};
use dm_value::{FieldName, Value};
use dm_world::allocate_world;

const MAX_EAGER_ARTIFACT_DIAGNOSTICS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Command {
    Compile,
    Plan,
    Boot,
    Sweep,
    SweepClosure,
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
    let mut arguments = env::args_os().skip(1);
    let Some(first) = arguments.next() else {
        eprintln!(
            "usage: dm-lifecycle [compile|plan|boot|sweep|sweep-closure] <world.dme> [map.dmm]"
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
    } else {
        (Command::Plan, PathBuf::from(first))
    };
    let requested_map = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        eprintln!(
            "usage: dm-lifecycle [compile|plan|boot|sweep|sweep-closure] <world.dme> [map.dmm]"
        );
        return ExitCode::from(2);
    }
    if command == Command::Compile && requested_map.is_some() {
        eprintln!("usage: dm-lifecycle compile <world.dme>");
        return ExitCode::from(2);
    }
    let compile_started = Instant::now();
    let cached_compilation = matches!(command, Command::Compile | Command::Boot);
    if cached_compilation {
        let progress = if command == Command::Compile {
            "compile-progress"
        } else {
            "boot-progress"
        };
        eprintln!(
            "{progress}: preparing compiled executable {}",
            environment.display()
        );
    }
    let cache_file = project_cache_file(&environment);
    let artifact_file = executable_artifact_file(&environment);
    let mut cached_executable = None;
    let mut cached_procedures = None;
    let compilation_result: Result<_, String> = if cached_compilation {
        prepare_compiled_executable(&environment, &cache_file, &artifact_file).map(|prepared| {
            let progress = if command == Command::Compile {
                "compile-progress"
            } else {
                "boot-progress"
            };
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
        let progress = if command == Command::Compile {
            "compile-progress"
        } else {
            "boot-progress"
        };
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
        eprintln!("boot-progress: loading map");
        let (map_path, map_source) = match load_map(&compilation, requested_map.as_deref()) {
            Ok(map) => map,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
        eprintln!("boot-progress: parsing map {map_path}");
        let map = match dm_map::parse(&map_source) {
            Ok(map) => map,
            Err(error) => {
                eprintln!("{map_path}: {error}");
                return ExitCode::FAILURE;
            }
        };
        let world = dm_world::build_plan(&map, &compilation);
        drop(map);
        drop(map_source);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
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
        RuntimeImage::from_compilation_with_prelinked_module(
            &compilation,
            &procedures,
            precompiled.module_mut_for_runtime_initializers(),
            runtime_observer,
        )
    } else {
        RuntimeImage::from_compilation_with_observer(&compilation, runtime_observer)
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
    if command == Command::Boot {
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
        let map = match dm_map::parse(&map_source) {
            Ok(map) => map,
            Err(error) => {
                eprintln!("{map_path}: {error}");
                return ExitCode::FAILURE;
            }
        };
        (map_path, dm_world::build_plan(&map, &compilation))
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
        let audit_runtime = env::var_os("DREAM64_BOOT_AUDIT_RUNTIME").is_some();
        let execution = match if audit_runtime {
            audit_initialization_plan_with_precompiled(
                &index,
                &plan,
                &allocation,
                &mut runtime,
                SchedulerDrainLimits::default(),
                precompiled,
            )
        } else {
            execute_boot_initialization_plan_with_precompiled(
                &index,
                &plan,
                &allocation,
                &mut runtime,
                SchedulerDrainLimits::default(),
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
        eprintln!("boot-progress: headless ready; entering persistent scheduler loop");
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
        loop {
            let slice_started = Instant::now();
            let tick_duration = precompiled.persistent_tick_duration();
            let scheduler = match advance_persistent_scheduler(
                precompiled,
                &mut runtime,
                SchedulerDrainLimits {
                    max_ticks: 1,
                    max_rounds: 10_000,
                },
            ) {
                Ok(scheduler) => scheduler,
                Err(error) => {
                    eprintln!("persistent scheduler: {error}");
                    return ExitCode::FAILURE;
                }
            };
            slices = slices.saturating_add(1);
            if slices == 1 || slices % 100 == 0 {
                eprintln!(
                    "server-progress: scheduler slice={} tick={} rounds={} completed={} failed={} pending={} termination={:?}",
                    slices,
                    scheduler.final_tick,
                    scheduler.rounds,
                    scheduler.completed_tasks,
                    scheduler.failed_tasks,
                    scheduler.pending_tasks,
                    scheduler.termination,
                );
            }
            if let Some(limit) = max_slices
                && slices >= limit
            {
                eprintln!("boot-progress: reached DREAM64_BOOT_MAX_SLICES={limit}; stopping");
                return ExitCode::SUCCESS;
            }
            if let Some(remaining) = tick_duration.checked_sub(slice_started.elapsed()) {
                std::thread::sleep(remaining);
            }
        }
    }
    ExitCode::SUCCESS
}

const COMPILATION_ARTIFACT_SECTION: u32 = 1;
const EXECUTABLE_ARTIFACT_SECTION: u32 = 2;

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
    // trust the heavyweight executable. On a warm boot this is the only
    // non-artifact project load and does not lex, parse, build an object tree,
    // lower a body, or construct runtime state.
    let (project, project_snapshot_hit) = Project::load_cached_exact(environment, cache_file)
        .map_err(|error| format!("project snapshot: {error}"))?;
    let project_fingerprint = *project.content_fingerprint().as_bytes();
    let miss_reason = if project_snapshot_hit {
        match decode_compiled_executable(artifact_file, project_fingerprint) {
            Ok((compilation, executable)) => {
                let procedures = ProcedureRegistry::build(&compilation);
                return Ok(PreparedCompiledExecutable {
                    compilation,
                    procedures,
                    executable,
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
    let (compilation, cache_stats) = CompilerDatabase::new()
        .compile_cached_with_stats(environment, cache_file)
        .map_err(|error| error.to_string())?;
    let procedures = ProcedureRegistry::build(&compilation);
    let executable = procedures
        .compile_vm_all_symbolic_deferred(&compilation)
        .map_err(|error| format!("complete executable lowering: {error}"))?;
    let executable = executable
        .into_fully_eager_bounded(MAX_EAGER_ARTIFACT_DIAGNOSTICS)
        .map_err(|error| format!("complete executable lowering: {error}"))?;
    if executable.module().deferred_procedure_count() != 0 {
        return Err("complete executable lowering retained deferred procedures".to_owned());
    }
    let new_lowerings = executable.module().procedure_count();
    let compilation_payload = compilation.encode_compiled_artifact();
    let executable_payload = executable.encode_compiled_artifact()?;
    eprintln!(
        "compile-progress: executable-artifact-payloads frontend_bytes={} executable_bytes={}",
        compilation_payload.len(),
        executable_payload.len(),
    );
    let fingerprint = *compilation.project().content_fingerprint().as_bytes();
    let artifact = CompiledArtifact::new(
        fingerprint,
        vec![
            ArtifactSection::new(COMPILATION_ARTIFACT_SECTION, compilation_payload),
            ArtifactSection::new(EXECUTABLE_ARTIFACT_SECTION, executable_payload),
        ],
    )
    .map_err(|error| format!("build executable artifact: {error}"))?;
    let write_stats = artifact
        .write_atomic_with_stats(artifact_file)
        .map_err(|error| format!("write executable artifact: {error}"))?;
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
) -> Result<(Compilation, ExecutableProcedures), String> {
    let artifact = CompiledArtifact::read_from(artifact_file, project_fingerprint)
        .map_err(|error| error.to_string())?;
    let storage = artifact.storage_stats();
    eprintln!(
        "boot-progress: executable-artifact-storage payload_bytes={} backing_bytes={} backing_allocations={}",
        storage.payload_bytes, storage.backing_bytes, storage.backing_allocations,
    );
    if artifact.sections().len() != 2 {
        return Err(format!(
            "compiled executable contains {} sections instead of 2",
            artifact.sections().len()
        ));
    }
    let compilation = Compilation::decode_compiled_artifact(
        artifact
            .section(COMPILATION_ARTIFACT_SECTION)
            .ok_or_else(|| "compiled executable is missing the frontend section".to_owned())?
            .payload(),
    )?;
    if compilation.project().content_fingerprint().as_bytes() != &project_fingerprint {
        return Err(
            "compiled executable frontend fingerprint disagrees with its envelope".to_owned(),
        );
    }
    let executable = ExecutableProcedures::decode_compiled_artifact(
        artifact
            .section(EXECUTABLE_ARTIFACT_SECTION)
            .ok_or_else(|| "compiled executable is missing the bytecode section".to_owned())?
            .payload(),
    )?;
    if executable.module().deferred_procedure_count() != 0 {
        return Err("compiled executable contains deferred procedures".to_owned());
    }
    Ok((compilation, executable))
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
        let original_modified = original_metadata
            .modified()
            .expect("source mtime should be readable");
        Fixture::write_source(&fixture.source, 2);
        fs::File::options()
            .write(true)
            .open(&fixture.source)
            .expect("changed source should open")
            .set_times(fs::FileTimes::new().set_modified(original_modified))
            .expect("original source timestamp should be restored");
        let changed_metadata =
            fs::metadata(&fixture.source).expect("changed metadata should exist");
        assert_eq!(changed_metadata.len(), original_metadata.len());
        assert_eq!(changed_metadata.modified().unwrap(), original_modified);
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
        let (_, left) = decode_compiled_executable(&fixture.artifact, fingerprint)
            .expect("first executable should load");
        let (_, right) = decode_compiled_executable(&fixture.artifact, fingerprint)
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
