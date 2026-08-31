//! Human-readable stdout/stderr reporting for `dream64-server`: plan and boot
//! summaries, the headless readiness probes derived from the compiled project,
//! runtime-initializer diagnostics, and opt-in ready-global inspection.

use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use dm_compiler::Compilation;
use dm_lifecycle::{HeadlessReadinessProbe, LifecycleIndex, LifecycleKind, LifecycleResolution};
use dm_runtime::{RuntimeImage, RuntimeInitializerDiagnostic};
use dm_semantics::ProcedureRegistry;
use dm_value::{FieldName, Value};

/// Prints selected live global datum fields at the authoritative-ready
/// boundary. This is deliberately opt-in: production servers retain the same
/// output and behavior, while compatibility investigations can inspect the
/// actual VM heap without adding game-specific knowledge to the engine.
///
/// Syntax: `DREAM64_READY_INSPECT=SSmapping:z_list,z_level_to_stack;Master:processing`.
pub(crate) fn inspect_ready_globals(state: &dm_vm::ExecutionState) {
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

pub(crate) fn format_runtime_diagnostic(diagnostic: &RuntimeInitializerDiagnostic) -> String {
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

pub(crate) fn master_controller_readiness(
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

pub(crate) fn lobby_pregame_readiness(
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

pub(crate) fn print_compatibility_sweep(sweep: &dm_lifecycle::LifecycleCompatibilitySweep) {
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

pub(crate) fn print_plan_summary(
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

pub(crate) fn print_boot_summary(
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

pub(crate) fn load_map(
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
