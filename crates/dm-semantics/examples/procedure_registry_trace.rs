use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use dm_compiler::{Compilation, CompilerDatabase};
use dm_semantics::ProcedureRegistry;

fn main() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    let Some(environment) = arguments.next() else {
        eprintln!("usage: procedure_registry_trace <world.dme> <procedure-path>...");
        return ExitCode::from(2);
    };
    let paths = arguments
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if paths.is_empty() {
        eprintln!("usage: procedure_registry_trace <world.dme> <procedure-path>...");
        return ExitCode::from(2);
    }

    let environment = PathBuf::from(environment);
    let compilation = match CompilerDatabase::new().compile(&environment) {
        Ok(compilation) => compilation,
        Err(error) => {
            eprintln!("{}: {error}", environment.display());
            return ExitCode::FAILURE;
        }
    };
    trace(&compilation, &paths);
    ExitCode::SUCCESS
}

fn trace(compilation: &Compilation, paths: &[String]) {
    let registry = ProcedureRegistry::build(compilation);
    for requested in paths {
        let Some(procedure) = registry
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == *requested)
        else {
            println!("trace procedure={requested} missing");
            continue;
        };
        println!(
            "trace procedure={} implementations={}",
            procedure.path,
            procedure.implementations.len()
        );
        for implementation in &procedure.implementations {
            let file = compilation
                .project()
                .file(implementation.file_id)
                .expect("implementation file should exist");
            let original = file.original_span(implementation.span);
            let line = file.text().ok().map_or(0, |source| {
                source[..original.start.min(source.len())].lines().count() + 1
            });
            println!(
                "trace implementation={} ordinal={} source={}:{} original_span={:?}",
                implementation.id.index(),
                implementation.ordinal,
                file.relative_path.display(),
                line,
                original
            );
            let definition = compilation
                .syntax(implementation.file_id)
                .and_then(|syntax| syntax.definitions.get(implementation.definition_index))
                .expect("implementation definition should exist");
            println!(
                "trace definition_path={} definition_kind={:?} header_tokens={:?}",
                definition.path, definition.kind, definition.header
            );
            for body_line in &definition.body {
                println!(
                    "trace body indent={} tokens={:?}",
                    body_line.indentation.spaces, body_line.tokens
                );
            }
        }
    }
}
