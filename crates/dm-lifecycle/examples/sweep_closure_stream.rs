use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use dm_compiler::{Compilation, CompilerDatabase};
use dm_lifecycle::{
    InitializationEvent, LifecycleCompatibilityLocation, LifecycleIndex, LifecycleResolution,
    build_initialization_plan,
};
use dm_runtime::RuntimeImage;
use dm_semantics::{ProcedureImplementationId, ProcedureRegistry};

#[allow(clippy::too_many_lines)]
fn main() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    let Some(environment) = arguments.next() else {
        eprintln!("usage: sweep_closure_stream <world.dme> [map.dmm]");
        return ExitCode::from(2);
    };
    let environment = PathBuf::from(environment);
    let requested_map = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        eprintln!("usage: sweep_closure_stream <world.dme> [map.dmm]");
        return ExitCode::from(2);
    }

    let compilation = match CompilerDatabase::new().compile(&environment) {
        Ok(compilation) => compilation,
        Err(error) => {
            eprintln!("{}: {error}", environment.display());
            return ExitCode::FAILURE;
        }
    };
    let runtime = match RuntimeImage::from_compilation(&compilation) {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("runtime image: {error}");
            return ExitCode::FAILURE;
        }
    };
    let procedures = ProcedureRegistry::build(&compilation);
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
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
    let world = dm_world::build_plan(&map, &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world, map_path);

    let mut locations =
        BTreeMap::<ProcedureImplementationId, Vec<LifecycleCompatibilityLocation>>::new();
    for event in &plan.events {
        let InitializationEvent::Lifecycle {
            kind, type_index, ..
        } = *event
        else {
            continue;
        };
        let Some(lifecycle) = index.types().get(type_index) else {
            continue;
        };
        let LifecycleResolution::Resolved(target) = lifecycle.targets.get(kind) else {
            continue;
        };
        let location = LifecycleCompatibilityLocation {
            kind,
            procedure_path: target.procedure_path.clone(),
            source: target.source.clone(),
        };
        let entry = locations.entry(target.implementation).or_default();
        if !entry.contains(&location) {
            entry.push(location);
        }
    }

    let targets = locations.len();
    println!("sweep_targets={targets}");
    let ordered_targets = locations.keys().copied().collect::<Vec<_>>();
    let (reachable, closure_stats) =
        procedures.implementation_closure_with_stats(&compilation, ordered_targets.iter().copied());
    let eager =
        procedures.eager_implementation_closure(&compilation, ordered_targets.iter().copied());
    let worker_count = env::var("DREAM64_SWEEP_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count != 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(4, usize::from)
                .min(8)
        });
    let reachable_count = reachable.len();
    let eager = eager.into_iter().collect::<Vec<_>>();
    let chunk_size = eager.len().div_ceil(worker_count);
    eprintln!(
        "sweep-progress: compiling {} eager symbolic bodies with {} workers",
        eager.len(),
        worker_count
    );
    let independent = std::thread::scope(|scope| {
        let handles = eager
            .chunks(chunk_size.max(1))
            .map(|chunk| {
                scope.spawn(|| {
                    procedures.compile_vm_bodies_independently(&compilation, chunk.iter().copied())
                })
            })
            .collect::<Vec<_>>();
        let mut results = Vec::with_capacity(eager.len());
        for handle in handles {
            results.extend(handle.join().expect("sweep worker should not panic"));
        }
        results
    });
    let body_results = independent
        .into_iter()
        .map(|(implementation, result)| {
            let result = result.map(|_| ()).map_err(|error| {
                let message = error.message;
                (compatibility_category(&message), message)
            });
            (implementation, result)
        })
        .collect::<BTreeMap<_, _>>();
    eprintln!("sweep-progress: linking exact boot symbolic module");
    let symbolic = procedures
        .compile_vm_implementations_symbolic_dynamic(&compilation, ordered_targets.iter().copied());
    println!("sweep_mode=exact-boot-symbolic-module-plus-parallel-eager-audit");
    println!("sweep_workers={worker_count}");
    println!("sweep_reachable_bodies={reachable_count}");
    println!("sweep_eager_bodies={}", body_results.len());
    println!(
        "sweep_deferred_symbols={}",
        reachable_count.saturating_sub(body_results.len())
    );
    match &symbolic {
        Ok(executable) => {
            println!("sweep_symbolic_module=compatible");
            println!(
                "sweep_symbolic_module_deferred={}",
                executable.module().deferred_procedure_count()
            );
            println!(
                "sweep_symbolic_module_materialized={}",
                executable.module().materialized_deferred_procedure_count()
            );
        }
        Err(error) => {
            println!("sweep_symbolic_module=blocked");
            println!(
                "sweep_symbolic_module_first_error category={:?} message={:?}",
                compatibility_category(&error.message),
                error.message
            );
        }
    }
    println!(
        "sweep_closure_bodies_visited={}",
        closure_stats.bodies_visited
    );
    println!(
        "sweep_closure_static_selectors={}",
        closure_stats.static_selectors_resolved
    );
    println!(
        "sweep_closure_dynamic_selectors={}",
        closure_stats.dynamic_selectors_resolved
    );
    println!(
        "sweep_closure_dynamic_candidates={}",
        closure_stats.dynamic_candidates_considered
    );
    let mut compatible = 0usize;
    let mut grouped = BTreeMap::<(String, String), Vec<LifecycleCompatibilityLocation>>::new();
    let trace_procedure = env::var("DREAM64_TRACE_PROCEDURE").ok();
    let only_procedure = env::var("DREAM64_SWEEP_ONLY_PROCEDURE").ok();

    let mut body_groups = BTreeMap::<(String, String), Vec<ProcedureImplementationId>>::new();
    for (implementation, result) in &body_results {
        if let Err((category, message)) = result {
            body_groups
                .entry((category.clone(), message.clone()))
                .or_default()
                .push(*implementation);
        }
    }
    println!("sweep_eager_issue_groups={}", body_groups.len());
    let eager_blocked = !body_groups.is_empty();
    for ((category, message), implementations) in &body_groups {
        println!(
            "sweep_eager_issue category={category:?} bodies={} message={message:?}",
            implementations.len()
        );
        for implementation in implementations.iter().take(5) {
            let procedure = procedures
                .procedure(implementation.procedure())
                .map_or_else(
                    || "<missing>".to_owned(),
                    |procedure| procedure.path.to_string(),
                );
            println!("sweep_eager_body procedure={procedure} implementation={implementation:?}");
        }
    }

    for (index, implementation) in ordered_targets.into_iter().enumerate() {
        if only_procedure.as_deref().is_some_and(|path| {
            procedures
                .procedure(implementation.procedure())
                .is_some_and(|procedure| procedure.path.to_string() != path)
        }) {
            continue;
        }
        if index == 0 || (index + 1) % 10 == 0 || index + 1 == targets {
            eprintln!("sweep-progress: target {}/{}", index + 1, targets);
        }

        if trace_procedure.as_deref().is_some_and(|path| {
            procedures
                .procedure(implementation.procedure())
                .is_some_and(|procedure| procedure.path.to_string() == path)
        }) {
            let body = procedures
                .implementation(implementation)
                .expect("sweep implementation should exist");
            let definition = compilation
                .syntax(body.file_id)
                .and_then(|syntax| syntax.definitions.get(body.definition_index))
                .expect("sweep implementation syntax should exist");
            eprintln!("sweep-trace procedure={trace_procedure:?}");
            for line in &definition.body {
                eprintln!(
                    "sweep-trace indent={} tokens={:?}",
                    line.indentation.spaces, line.tokens
                );
            }
        }
        let result = body_results
            .get(&implementation)
            .and_then(|result| match result {
                Err(error) => Some(error.clone()),
                Ok(()) => None,
            });
        let target_locations = locations
            .remove(&implementation)
            .expect("sweep target should retain its locations");
        match result {
            None => {
                compatible += 1;
            }
            Some((category, message)) => {
                grouped
                    .entry((category, message))
                    .or_default()
                    .extend(target_locations);
            }
        }
    }

    println!("sweep_direct_compatible={compatible}");
    println!("sweep_issue_groups={}", grouped.len());
    let direct_blocked = !grouped.is_empty();
    for ((category, message), mut issue_locations) in grouped {
        issue_locations.sort_by(|left, right| {
            (
                left.source.path.as_str(),
                left.source.span.start,
                left.procedure_path.as_str(),
                left.kind,
            )
                .cmp(&(
                    right.source.path.as_str(),
                    right.source.span.start,
                    right.procedure_path.as_str(),
                    right.kind,
                ))
        });
        println!(
            "sweep_issue category={category:?} locations={} message={message:?}",
            issue_locations.len()
        );
        for location in issue_locations {
            println!(
                "sweep_location phase={:?} procedure={} source={}:{}",
                location.kind,
                location.procedure_path,
                location.source.path,
                location.source.span.start
            );
        }
    }

    if symbolic.is_err() || eager_blocked || direct_blocked {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn compatibility_category(message: &str) -> String {
    message
        .split_once(':')
        .map_or_else(|| message.to_owned(), |(category, _)| category.to_owned())
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
