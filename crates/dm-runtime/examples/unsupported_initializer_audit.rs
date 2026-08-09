use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;

use dm_runtime::RuntimeImage;

fn main() -> ExitCode {
    let Some(project) = env::args_os().nth(1) else {
        eprintln!("usage: unsupported_initializer_audit <project.dme>");
        return ExitCode::from(2);
    };
    let image = match RuntimeImage::load(project) {
        Ok(image) => image,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let mut groups = BTreeMap::<(String, String), Vec<_>>::new();
    for diagnostic in image.diagnostics() {
        groups
            .entry((
                format!("{:?}", diagnostic.phase),
                diagnostic.message.clone(),
            ))
            .or_default()
            .push(diagnostic);
    }
    println!("unsupported_initializers={}", image.diagnostics().len());
    println!("unsupported_groups={}", groups.len());
    for ((phase, message), diagnostics) in groups {
        println!(
            "group phase={phase} count={} message={message:?}",
            diagnostics.len()
        );
        for diagnostic in &diagnostics {
            println!(
                "  variable={} source={} span={}..{}",
                diagnostic.variable_path,
                diagnostic.source_path,
                diagnostic.blocker_span.start,
                diagnostic.blocker_span.end,
            );
        }
    }
    ExitCode::SUCCESS
}
