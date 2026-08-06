use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;

use dm_globals::UnsupportedCategory;
use dm_runtime::RuntimeImage;

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(project) = arguments.next() else {
        eprintln!("usage: {} <project.dme>", program.to_string_lossy());
        return ExitCode::from(2);
    };
    if arguments.next().is_some() {
        eprintln!("usage: {} <project.dme>", program.to_string_lossy());
        return ExitCode::from(2);
    }

    let image = match RuntimeImage::load(&project) {
        Ok(image) => image,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let stats = image.stats();
    println!("variables inventoried:      {}", stats.variables);
    println!("initializer steps:          {}", stats.initializer_steps);
    println!(
        "constants materialized:     {}",
        stats.constants_materialized
    );
    println!(
        "dynamic initializers:       {}",
        stats.dynamic_initializers_materialized
    );
    println!("global/static slots:        {}", stats.runtime_variables);
    println!("runtime types:              {}", stats.runtime_types);
    println!("direct default layers:      {}", stats.default_layers);
    println!("constant lists:             {}", stats.constant_lists);
    println!(
        "unsupported initializers:  {}",
        stats.unsupported_initializers
    );
    println!(
        "datums allocated:           {}",
        image.heap().datums().count()
    );

    let mut unsupported = BTreeMap::<UnsupportedCategory, usize>::new();
    for diagnostic in image.diagnostics() {
        *unsupported.entry(diagnostic.category).or_default() += 1;
    }
    for (category, count) in unsupported {
        println!("  {category:?}: {count}");
    }
    ExitCode::SUCCESS
}
