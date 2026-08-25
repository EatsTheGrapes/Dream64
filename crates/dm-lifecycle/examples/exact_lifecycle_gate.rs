use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use dm_compiler::CompilerDatabase;
use dm_lifecycle::{
    LifecycleIndex, LifecycleKind, LifecycleResolution, precompile_lifecycle_for_world,
};
use dm_semantics::{ProcedureImplementationId, ProcedureRegistry};
use dm_world::InitializerResolution;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    let environment = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: exact_lifecycle_gate <world.dme> <map.dmm>")?,
    );
    let map_path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: exact_lifecycle_gate <world.dme> <map.dmm>")?,
    );
    if arguments.next().is_some() {
        return Err("usage: exact_lifecycle_gate <world.dme> <map.dmm>".to_owned());
    }
    let compilation = CompilerDatabase::new()
        .compile(&environment)
        .map_err(|error| format!("{}: {error}", environment.display()))?;
    let procedures = ProcedureRegistry::build(&compilation);
    if let Ok(trace_path) = env::var("DREAM64_GATE_TRACE_ONLY") {
        let procedure = procedures
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == trace_path)
            .ok_or_else(|| format!("trace procedure {trace_path:?} is absent"))?;
        for implementation in &procedure.implementations {
            let definition = compilation
                .syntax(implementation.file_id)
                .and_then(|syntax| syntax.definitions.get(implementation.definition_index))
                .ok_or_else(|| "trace implementation syntax is absent".to_owned())?;
            for (index, line) in definition.body.iter().enumerate() {
                println!(
                    "trace_line index={index} indent={:?} tokens={:?}",
                    line.indentation, line.tokens
                );
            }
        }
        return Ok(());
    }
    let index = LifecycleIndex::build_compile_only(&compilation, &procedures);
    let map_source = fs::read_to_string(&map_path)
        .map_err(|error| format!("{}: {error}", map_path.display()))?;
    let map =
        dm_map::parse(&map_source).map_err(|error| format!("{}: {error}", map_path.display()))?;
    let world = dm_world::build_plan(&map, &compilation);

    let mut roots = BTreeSet::new();
    if let Some(world_type) = index.find_path("/world") {
        insert_target(world_type, LifecycleKind::Genesis, &mut roots);
        insert_target(world_type, LifecycleKind::New, &mut roots);
    }
    let map_paths = world
        .templates()
        .values()
        .flat_map(|template| &template.initializers)
        .filter_map(|initializer| match initializer.resolution {
            InitializerResolution::Resolved { .. } => Some(initializer.path.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    for path in map_paths {
        if let Some(lifecycle) = index.find_path(path) {
            for kind in [
                LifecycleKind::New,
                LifecycleKind::Initialize,
                LifecycleKind::LateInitialize,
            ] {
                insert_target(lifecycle, kind, &mut roots);
            }
        }
    }
    println!("gate_targets={}", roots.len());
    let (reachable, stats) =
        procedures.implementation_closure_with_stats(&compilation, roots.iter().copied());
    let eager = procedures.eager_implementation_closure(&compilation, roots.iter().copied());
    println!("gate_closure_bodies={}", reachable.len());
    println!("gate_eager_bodies={}", eager.len());
    println!("gate_static_selectors={}", stats.static_selectors_resolved);
    println!(
        "gate_dynamic_selectors={}",
        stats.dynamic_selectors_resolved
    );

    let mut issues = BTreeMap::<String, Vec<ProcedureImplementationId>>::new();
    for (implementation, result) in
        procedures.compile_vm_bodies_independently(&compilation, eager.iter().copied())
    {
        if let Err(error) = result {
            issues
                .entry(error.message)
                .or_default()
                .push(implementation);
        }
    }
    println!("gate_issue_groups={}", issues.len());
    for (message, implementations) in &issues {
        println!(
            "gate_issue bodies={} message={message:?}",
            implementations.len()
        );
        for implementation in implementations.iter().take(8) {
            let path = procedures
                .procedure(implementation.procedure())
                .map_or_else(
                    || "<missing>".to_owned(),
                    |procedure| procedure.path.to_string(),
                );
            println!("gate_issue_body procedure={path} implementation={implementation:?}");
        }
    }
    let mut deferred_issues = BTreeMap::<String, Vec<ProcedureImplementationId>>::new();
    if env::var_os("DREAM64_GATE_VALIDATE_DEFERRED").is_some() {
        let mut deferred = reachable.difference(&eager).copied().collect::<Vec<_>>();
        if env::var_os("DREAM64_GATE_DEFERRED_STARTUP_ONLY").is_some() {
            deferred.retain(|implementation| {
                procedures
                    .procedure(implementation.procedure())
                    .is_some_and(|procedure| startup_family(&procedure.path.to_string()) != "other")
            });
            println!("gate_deferred_validation_filter=startup");
        }
        println!("gate_deferred_validation_bodies={}", deferred.len());
        // Compile each retained deferred body independently and immediately
        // discard its Program. This validates runtime-reachable lazy bodies
        // without materializing all of them into the production Module or
        // increasing its steady-state memory footprint.
        for (implementation, result) in
            procedures.compile_vm_bodies_independently(&compilation, deferred)
        {
            if let Err(error) = result {
                deferred_issues
                    .entry(error.message)
                    .or_default()
                    .push(implementation);
            }
        }
        println!("gate_deferred_issue_groups={}", deferred_issues.len());
        for (message, implementations) in &deferred_issues {
            println!(
                "gate_deferred_issue bodies={} message={message:?}",
                implementations.len()
            );
            for implementation in implementations.iter().take(8) {
                let path = procedures
                    .procedure(implementation.procedure())
                    .map_or_else(
                        || "<missing>".to_owned(),
                        |procedure| procedure.path.to_string(),
                    );
                println!(
                    "gate_deferred_issue_body procedure={path} implementation={implementation:?}"
                );
            }
        }
        if let Some(output) = env::var_os("DREAM64_GATE_DEFERRED_REPORT") {
            let mut report = String::from("category\tstartup_family\tbodies\tprocedure\tmessage\n");
            for (message, implementations) in &deferred_issues {
                let category = deferred_issue_category(message);
                for implementation in implementations {
                    let path = procedures
                        .procedure(implementation.procedure())
                        .map_or_else(
                            || "<missing>".to_owned(),
                            |procedure| procedure.path.to_string(),
                        );
                    let startup = startup_family(&path);
                    report.push_str(&format!(
                        "{category}\t{startup}\t1\t{}\t{}\n",
                        path.replace('\t', " "),
                        message.replace(['\t', '\n', '\r'], " "),
                    ));
                }
            }
            fs::write(&output, report).map_err(|error| {
                format!(
                    "failed to write deferred report {}: {error}",
                    PathBuf::from(output).display()
                )
            })?;
        }
    }
    // Keep the final compatibility decision on the exact production path.
    // The issue inventory above deliberately compiles eager bodies separately
    // so it can report more than the first failure, but boot uses this API to
    // select roots and build its symbolic module before RuntimeImage exists.
    let precompiled = precompile_lifecycle_for_world(&compilation, &procedures, &index, &world)
        .map_err(|error| format!("gate_precompile=blocked message={:?}", error.message))?;
    println!(
        "gate_precompile=compatible targets={} bodies={} procedures={} deferred={}",
        precompiled.targets(),
        precompiled.reachable_bodies(),
        precompiled.module_procedures(),
        precompiled.deferred_procedures(),
    );
    if issues.is_empty() && deferred_issues.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} eager and {} deferred lifecycle issue groups",
            issues.len(),
            deferred_issues.len(),
        ))
    }
}

fn deferred_issue_category(message: &str) -> &'static str {
    if message.contains(": unknown local") {
        "engine-field"
    } else if message.contains(": unknown procedure") {
        "builtin-or-link"
    } else if message.starts_with("unknown declared type") {
        "declaration-parse"
    } else if message.contains("received") || message.contains("expected") {
        "arity-or-syntax"
    } else if message.contains("unexpected token") || message.contains("unsupported statement") {
        "syntax"
    } else if message.starts_with("cannot assign") {
        "type-check"
    } else {
        "other"
    }
}

fn startup_family(path: &str) -> &'static str {
    if path.starts_with("/world/proc/Genesis") || path.starts_with("/world/proc/New") {
        "world"
    } else if path.contains("/datum/controller/global_vars/proc/") {
        "glob"
    } else if path.contains("/datum/controller/master/proc/") {
        "master"
    } else if path.ends_with("/proc/PreInit") {
        "preinit"
    } else if path.ends_with("/proc/Initialize") {
        "initialize"
    } else if path.ends_with("/proc/Destroy") {
        "destroy"
    } else {
        "other"
    }
}

fn insert_target(
    lifecycle: &dm_lifecycle::TypeLifecycle,
    kind: LifecycleKind,
    roots: &mut BTreeSet<ProcedureImplementationId>,
) {
    if let LifecycleResolution::Resolved(target) = lifecycle.targets.get(kind) {
        roots.insert(target.implementation);
    }
}
