use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::CompilerDatabase;

fn main() -> ExitCode {
    let Some(environment) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: cargo run -p dm-compiler --example frontend_check -- <project.dme>");
        return ExitCode::from(2);
    };
    let started = Instant::now();
    let compilation = match CompilerDatabase::new().compile(&environment) {
        Ok(compilation) => compilation,
        Err(error) => {
            eprintln!("{}: {error}", environment.display());
            return ExitCode::FAILURE;
        }
    };
    let stats = compilation.stats();
    println!(
        "files={} parsed={} bytes={} definitions={} nodes={} declarations={}",
        stats.project_files,
        stats.parsed_files,
        stats.project_bytes,
        stats.definitions,
        stats.code_nodes,
        stats.code_declarations
    );
    println!(
        "notes={} warnings={} errors={} elapsed-ms={}",
        stats.notes,
        stats.warnings,
        stats.errors,
        started.elapsed().as_millis()
    );
    for diagnostic in compilation.diagnostics().iter().take(20) {
        let location = diagnostic.location.as_ref().map_or_else(
            || "<project>".to_owned(),
            |location| {
                location.span.map_or_else(
                    || location.path.display().to_string(),
                    |span| format!("{}:{}..{}", location.path.display(), span.start, span.end),
                )
            },
        );
        println!(
            "{:?} {:?} {location}: {}",
            diagnostic.severity, diagnostic.kind, diagnostic.message
        );
    }
    if stats.errors == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
