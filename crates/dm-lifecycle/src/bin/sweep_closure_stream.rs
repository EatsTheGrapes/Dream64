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
    let mut compatible = 0usize;
    let mut grouped = BTreeMap::<(String, String), Vec<LifecycleCompatibilityLocation>>::new();

    for (index, implementation) in ordered_targets.into_iter().enumerate() {
        if index == 0 || (index + 1) % 10 == 0 || index + 1 == targets {
            eprintln!("sweep-progress: target {}/{}", index + 1, targets);
        }

        let result = procedures.compile_vm_implementations(&compilation, [implementation]);
        let target_locations = locations
            .remove(&implementation)
            .expect("sweep target should retain its locations");
        match result {
            Ok(executable) => {
                compatible += 1;
                // Drop each potentially large overlapping closure before compiling
                // the next one so the audit's live memory is bounded by one closure.
                drop(executable);
            }
            Err(error) => {
                let message = error.message;
                let category = compatibility_category(&message);
                grouped
                    .entry((category, message))
                    .or_default()
                    .extend(target_locations);
            }
        }
    }

    println!("sweep_compatible={compatible}");
    println!("sweep_issue_groups={}", grouped.len());
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

    ExitCode::SUCCESS
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
