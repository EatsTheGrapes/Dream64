use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use dm_compiler::CompilerDatabase;
use dm_globals::{InitializerClass, VariableRegistry};

fn main() -> ExitCode {
    let Some(project_path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: dm-globals <world.dme>");
        return ExitCode::FAILURE;
    };
    let compilation = match CompilerDatabase::new().compile(&project_path) {
        Ok(compilation) => compilation,
        Err(error) => {
            eprintln!("failed to compile {}: {error}", project_path.display());
            return ExitCode::FAILURE;
        }
    };
    let registry = VariableRegistry::build(&compilation);
    let initialized = registry
        .entries()
        .iter()
        .filter(|entry| entry.initializer.is_some())
        .count();
    let constant_safe = registry
        .entries()
        .iter()
        .filter(|entry| {
            entry
                .initializer
                .as_ref()
                .is_some_and(|initializer| initializer.class == InitializerClass::ConstantSafe)
        })
        .count();

    println!("variables={}", registry.entries().len());
    println!("initialized={initialized}");
    println!("constant_safe={constant_safe}");
    for (storage, count) in registry.storage_counts() {
        println!("storage_{storage:?}={count}");
    }
    let mut blockers: Vec<_> = registry.runtime_blocker_counts().into_iter().collect();
    blockers.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    for (blocker, count) in blockers {
        println!("runtime_{blocker:?}={count}");
    }
    for (shape, count) in registry.constant_value_counts() {
        println!("constant_{shape:?}={count}");
    }
    let mut unsupported: Vec<_> = registry.unsupported_constant_counts().into_iter().collect();
    unsupported.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    for (category, count) in unsupported {
        println!("unsupported_{category:?}={count}");
    }
    let plans = registry.initialization_plans();
    println!("plan_global_steps={}", plans.global_steps.len());
    println!("plan_type_count={}", plans.type_defaults.len());
    println!(
        "plan_type_steps={}",
        plans
            .type_defaults
            .iter()
            .map(|plan| plan.steps.len())
            .sum::<usize>()
    );
    ExitCode::SUCCESS
}
