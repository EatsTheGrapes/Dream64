use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use dm_compiler::{Compilation, CompilerDatabase};
use dm_object_tree::NodeKind;
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
        if let Some(path) = requested.strip_prefix("closure:") {
            let Some(procedure) = registry
                .procedures()
                .iter()
                .find(|procedure| procedure.path.to_string() == path)
            else {
                println!("trace closure procedure={path} missing");
                continue;
            };
            let Some(entry) = procedure.effective_target else {
                println!("trace closure procedure={path} has_no_body");
                continue;
            };
            let (closure, stats) = registry.implementation_closure_with_stats(compilation, [entry]);
            println!(
                "trace closure procedure={path} bodies={} visited={} static_selectors={} dynamic_selectors={} dynamic_candidates={}",
                closure.len(),
                stats.bodies_visited,
                stats.static_selectors_resolved,
                stats.dynamic_selectors_resolved,
                stats.dynamic_candidates_considered,
            );
            match registry.compile_vm_implementations(compilation, [entry]) {
                Ok(executable) => println!(
                    "trace closure compile=compatible module_procedures={} src_bindings={} global_bindings={}",
                    executable.stats().procedures,
                    executable.stats().src_field_bindings,
                    executable.stats().global_field_bindings,
                ),
                Err(error) => println!("trace closure compile=blocked error={:?}", error.message),
            }
            let issues = registry
                .compile_vm_bodies_independently(compilation, closure)
                .into_iter()
                .filter_map(|(implementation, result)| {
                    result.err().map(|error| (implementation, error.message))
                })
                .collect::<Vec<_>>();
            println!("trace closure body_issues={}", issues.len());
            for (implementation, message) in issues {
                println!(
                    "trace closure body_issue implementation={implementation:?} error={message:?}"
                );
            }
            continue;
        }
        if let Some(needle) = requested.strip_prefix("text:") {
            let mut visited = std::collections::BTreeSet::new();
            for segment in compilation.project().expansion_segments() {
                if !visited.insert(segment.file_id) {
                    continue;
                }
                let Some(file) = compilation.project().file(segment.file_id) else {
                    continue;
                };
                let Ok(source) = file.compiler_text() else {
                    continue;
                };
                for (line, text) in source
                    .lines()
                    .enumerate()
                    .filter(|(_, text)| text.contains(needle))
                {
                    println!(
                        "trace text source={}:{} value={text:?}",
                        file.relative_path.display(),
                        line + 1
                    );
                }
            }
            continue;
        }
        if let Some(variable_name) = requested.strip_prefix("var:") {
            for node in compilation.code_tree().nodes().iter().filter(|node| {
                node.kind == NodeKind::Variable
                    && node
                        .path
                        .segments()
                        .last()
                        .is_some_and(|name| name == variable_name)
            }) {
                println!(
                    "trace variable={} path={} owner_type={:?} declarations={} implicit={}",
                    variable_name,
                    node.path,
                    node.owner_type,
                    node.declarations.len(),
                    node.is_implicit()
                );
            }
            continue;
        }
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
