use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use dm_compiler::CompilerDatabase;
use dm_lifecycle::{LifecycleIndex, LifecycleKind, LifecycleResolution};
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
    let map_nodes = world
        .templates()
        .values()
        .flat_map(|template| &template.initializers)
        .filter_map(|initializer| match initializer.resolution {
            InitializerResolution::Resolved { node, .. } => Some(node),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    for node in map_nodes {
        if let Some(lifecycle) = index.find_node(node) {
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
    procedures
        .compile_vm_implementations_symbolic_dynamic(&compilation, roots)
        .map_err(|error| format!("gate_symbolic_module=blocked message={:?}", error.message))?;
    println!("gate_symbolic_module=compatible");
    if issues.is_empty() {
        Ok(())
    } else {
        Err(format!("{} eager lifecycle issue groups", issues.len()))
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
