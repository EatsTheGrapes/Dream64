use std::env;
use std::process::ExitCode;

use dm_lowering::lower_project;

fn main() -> ExitCode {
    let Some(environment) = env::args_os().nth(1) else {
        eprintln!("usage: cargo run --example inventory -- <project.dme> [top-count]");
        return ExitCode::from(2);
    };
    let top_count = env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let inventory = match lower_project(environment) {
        Ok(inventory) => inventory,
        Err(error) => {
            eprintln!("project load failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let stats = inventory.stats();
    println!("procedures: {}", stats.procedures);
    println!("implementations attempted: {}", stats.implementations);
    println!("lowered: {}", stats.lowered);
    println!("parent-linked: {}", stats.parent_linked);
    println!("unsupported: {}", stats.unsupported);
    println!("top blockers:");
    for blocker in inventory.blockers().iter().take(top_count) {
        println!(
            "  {:>7}  {:<20} {}",
            blocker.count,
            blocker.category.label(),
            blocker.example_message
        );
    }
    println!("first diagnostics:");
    for diagnostic in inventory.diagnostics().iter().take(top_count) {
        println!(
            "  {}:{}..{} {} [{}] {}",
            diagnostic.source_path,
            diagnostic.original_span.start,
            diagnostic.original_span.end,
            diagnostic.procedure_path,
            diagnostic.category.label(),
            diagnostic.message
        );
    }
    ExitCode::SUCCESS
}
