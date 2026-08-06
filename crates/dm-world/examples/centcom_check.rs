use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use dm_compiler::CompilerDatabase;
use dm_world::{AtomCategory, InitializerResolution, build_plan};

fn main() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    let Some(environment) = arguments.next().map(PathBuf::from) else {
        eprintln!("usage: cargo run --example centcom_check -- <project.dme> [CentCom.dmm]");
        return ExitCode::from(2);
    };
    let requested_map = arguments.next().map(PathBuf::from);
    let compilation = match CompilerDatabase::new().compile(&environment) {
        Ok(compilation) => compilation,
        Err(error) => {
            eprintln!("{}: {error}", environment.display());
            return ExitCode::FAILURE;
        }
    };
    let (map_name, source) = match load_map(&compilation, requested_map.as_deref()) {
        Ok(map) => map,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let map = match dm_map::parse(&source) {
        Ok(map) => map,
        Err(error) => {
            eprintln!("{map_name}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let plan = build_plan(&map, &compilation);
    let stats = plan.stats();
    let categories = category_counts(&plan);
    println!("map: {map_name}");
    println!(
        "templates={} cells={} initializers={} resolved={} placements={} resolved-placements={} diagnostics={}",
        stats.templates,
        stats.cells,
        stats.initializers,
        stats.resolved_initializers,
        stats.initializer_placements,
        stats.resolved_atom_placements,
        stats.diagnostics
    );
    println!(
        "categories: area={} turf={} movable={} other={}",
        categories.get(&AtomCategory::Area).copied().unwrap_or(0),
        categories.get(&AtomCategory::Turf).copied().unwrap_or(0),
        categories.get(&AtomCategory::Movable).copied().unwrap_or(0),
        categories
            .get(&AtomCategory::OtherType)
            .copied()
            .unwrap_or(0)
    );
    let mut diagnostic_counts = BTreeMap::new();
    for diagnostic in plan.diagnostics() {
        *diagnostic_counts.entry(diagnostic.kind).or_insert(0usize) += 1;
    }
    for (kind, count) in diagnostic_counts {
        println!("diagnostic {}={count}", kind.label());
    }
    for diagnostic in plan.diagnostics().iter().take(20) {
        println!(
            "  {}..{} {}{}",
            diagnostic.span.start,
            diagnostic.span.end,
            diagnostic.message,
            diagnostic
                .coordinate
                .map_or_else(String::new, |coordinate| format!(
                    " at ({},{},{})",
                    coordinate.x, coordinate.y, coordinate.z
                ))
        );
    }
    if plan.diagnostics().is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
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

fn category_counts(plan: &dm_world::WorldPlan) -> BTreeMap<AtomCategory, usize> {
    let mut counts = BTreeMap::new();
    for initializer in plan
        .templates()
        .values()
        .flat_map(|template| &template.initializers)
    {
        if let InitializerResolution::Resolved { category, .. } = initializer.resolution {
            *counts.entry(category).or_insert(0usize) += 1;
        }
    }
    counts
}
