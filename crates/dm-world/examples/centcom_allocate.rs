use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use dm_compiler::CompilerDatabase;
use dm_runtime::RuntimeImage;
use dm_world::{WorldAllocationWorkKind, allocate_world, build_plan};

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
    let environment = arguments.next().map(PathBuf::from).ok_or_else(|| {
        "usage: cargo run --example centcom_allocate -- <project.dme> [CentCom.dmm]".to_owned()
    })?;
    let requested_map = arguments.next().map(PathBuf::from);
    let compilation = CompilerDatabase::new()
        .compile(&environment)
        .map_err(|error| format!("{}: {error}", environment.display()))?;
    let (map_name, source) = load_map(&compilation, requested_map.as_deref())?;
    let map = dm_map::parse(&source).map_err(|error| format!("{map_name}: {error}"))?;
    let plan = build_plan(&map, &compilation);
    let mut image = RuntimeImage::from_compilation(&compilation)
        .map_err(|error| format!("runtime image: {error}"))?;
    let allocation =
        allocate_world(&plan, &mut image).map_err(|error| format!("world allocation: {error}"))?;
    let plan_stats = plan.stats();
    let runtime_stats = image.stats();
    let allocation_stats = allocation.stats();

    println!("map: {map_name}");
    println!(
        "plan: cells={} initializers={} resolved={} placements={} diagnostics={}",
        plan_stats.cells,
        plan_stats.initializers,
        plan_stats.resolved_initializers,
        plan_stats.initializer_placements,
        plan_stats.diagnostics,
    );
    println!(
        "runtime: types={} globals={} constants={} dynamic={} unsupported={}",
        runtime_stats.runtime_types,
        runtime_stats.runtime_variables,
        runtime_stats.constants_materialized,
        runtime_stats.dynamic_initializers_materialized,
        runtime_stats.unsupported_initializers,
    );
    println!(
        "allocation: datums={} areas={} turfs={} movables={} constant-overrides={} unsupported-overrides={} skipped={}",
        allocation_stats.datums_allocated,
        allocation_stats.unique_areas,
        allocation_stats.turfs,
        allocation_stats.movables,
        allocation_stats.constant_overrides,
        allocation_stats.unsupported_overrides,
        allocation_stats.skipped_initializers,
    );

    let mut work_counts = BTreeMap::new();
    for item in allocation.work_items() {
        *work_counts.entry(item.kind).or_insert(0usize) += 1;
    }
    for (kind, count) in work_counts {
        println!("work {}={count}", work_kind_label(kind));
    }
    Ok(())
}

fn load_map(
    compilation: &dm_compiler::Compilation,
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
        .ok_or_else(|| "compiled project contains no CentCom.dmm include".to_owned())?;
    let source = file
        .text()
        .map_err(|error| format!("{}: {error}", file.relative_path.display()))?;
    Ok((file.relative_path.display().to_string(), source.to_owned()))
}

const fn work_kind_label(kind: WorldAllocationWorkKind) -> &'static str {
    match kind {
        WorldAllocationWorkKind::UnknownInitializer => "unknown-initializer",
        WorldAllocationWorkKind::NonTypeInitializer => "non-type-initializer",
        WorldAllocationWorkKind::OtherType => "other-type",
        WorldAllocationWorkKind::MissingTemplate => "missing-template",
        WorldAllocationWorkKind::ExtraArea => "extra-area",
        WorldAllocationWorkKind::ExtraTurf => "extra-turf",
        WorldAllocationWorkKind::InvalidFieldName => "invalid-field-name",
        WorldAllocationWorkKind::DynamicOverride(_) => "dynamic-override",
    }
}
