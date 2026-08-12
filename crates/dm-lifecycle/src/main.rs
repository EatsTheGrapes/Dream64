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
    audit_initialization_plan_with_precompiled, build_initialization_plan,
    execute_initialization_plan_with_precompiled, precompile_lifecycle_for_world,
    sweep_lifecycle_compatibility, sweep_lifecycle_compatibility_with_closures,
};
use dm_runtime::{RuntimeImage, RuntimeImageConstructionEvent, RuntimeInitializerDiagnostic};
use dm_semantics::ProcedureRegistry;
use dm_value::{FieldName, Value};
use dm_world::allocate_world;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Command {
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
        eprintln!("usage: dm-lifecycle [plan|boot|sweep|sweep-closure] <world.dme> [map.dmm]");
        return ExitCode::from(2);
    };
    let (command, environment) = if first.as_os_str() == OsStr::new("plan") {
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
        eprintln!("usage: dm-lifecycle [plan|boot|sweep|sweep-closure] <world.dme> [map.dmm]");
        return ExitCode::from(2);
    }
    let compile_started = Instant::now();
    if command == Command::Boot {
        eprintln!("boot-progress: compiling project {}", environment.display());
    }
    let cache_file = project_cache_file(&environment);
    let compilation_result = if command == Command::Boot {
        CompilerDatabase::new()
            .compile_cached(&environment, &cache_file)
            .map(|(compilation, cache_hit)| (compilation, Some(cache_hit)))
    } else {
        CompilerDatabase::new()
            .compile(&environment)
            .map(|compilation| (compilation, None))
    };
    let (compilation, project_cache_hit) = match compilation_result {
        Ok(compilation) => compilation,
        Err(error) => {
            eprintln!("{}: {error}", environment.display());
            return ExitCode::FAILURE;
        }
    };
    if command == Command::Boot {
        eprintln!(
            "boot-progress: project-compile-complete elapsed_ms={} preprocessing_cache={} cache={}",
            compile_started.elapsed().as_millis(),
            if project_cache_hit == Some(true) {
                "hit"
            } else {
                "miss"
            },
            cache_file.display(),
        );
    }
    let procedures = ProcedureRegistry::build(&compilation);
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
        let precompiled =
            match precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
            {
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
    let mut runtime = match RuntimeImage::from_compilation_with_observer(
        &compilation,
        |event: RuntimeImageConstructionEvent| {
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
        },
    ) {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("runtime image: {error}");
            return ExitCode::FAILURE;
        }
    };
    if command == Command::Boot {
        let stats = runtime.stats();
        eprintln!(
            "boot-progress: initializer frontier selectors={} typed_constructors={} dynamic_constructor_fallback={} complete_inventory_fallback={} module_procedures={} deferred={} materialized={}",
            stats.initializer_frontier_selectors,
            stats.initializer_typed_constructor_targets,
            stats.initializer_dynamic_constructor_frontier,
            stats.initializer_complete_symbol_inventory,
            stats.initializer_module_procedures,
            stats.initializer_module_deferred_procedures,
            stats.initializer_module_materialized_procedures,
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
            execute_initialization_plan_with_precompiled(
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
                    "server-progress: scheduler slice={} tick={} rounds={} completed={} pending={} termination={:?}",
                    slices,
                    scheduler.final_tick,
                    scheduler.rounds,
                    scheduler.completed_tasks,
                    scheduler.pending_tasks,
                    scheduler.termination,
                );
            }
            if let Some(remaining) = tick_duration.checked_sub(slice_started.elapsed()) {
                std::thread::sleep(remaining);
            }
        }
    }
    ExitCode::SUCCESS
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
