use std::env;
use std::fs;
use std::process::ExitCode;

use dm_dmf::{DiagnosticSeverity, Section, parse};

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: cargo run --example check -- <skin.dmf>");
        return ExitCode::from(2);
    };
    let source = match fs::read_to_string(&path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("{}: {error}", path.to_string_lossy());
            return ExitCode::from(2);
        }
    };
    let document = parse(&source);
    let elements: usize = document
        .sections
        .iter()
        .map(|section| match section {
            Section::Window(window) => window.controls.len(),
            Section::Menu(menu) => menu.entries.len(),
            Section::MacroSet(macros) => macros.macros.len(),
        })
        .sum();
    println!(
        "sections={} elements={} comments={} diagnostics={}",
        document.sections.len(),
        elements,
        document.comments.len(),
        document.diagnostics.len()
    );
    for diagnostic in &document.diagnostics {
        eprintln!(
            "{}:{}..{}: {}",
            path.to_string_lossy(),
            diagnostic.span.start,
            diagnostic.span.end,
            diagnostic.message
        );
    }
    if document
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
