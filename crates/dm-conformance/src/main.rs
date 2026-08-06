//! Command-line entry point for reference compiler and lexer probes.

use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{self, Child, Command, ExitCode, Stdio};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

fn main() -> ExitCode {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    match run(&arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: &[OsString]) -> Result<(), String> {
    let Some(command) = arguments.first().and_then(|value| value.to_str()) else {
        return Err(usage());
    };
    match command {
        "check" => run_project_syntax_check(&arguments[1..]),
        "compile-check" => run_project_compile_check(&arguments[1..]),
        "execute" => run_procedure(&arguments[1..]),
        "frontend" => run_frontend(&arguments[1..]),
        "lex" => run_lexer(&arguments[1..]),
        "probe" => run_compiler_probe(&arguments[1..]),
        "project" => run_project(&arguments[1..]),
        "syntax" => run_syntax(&arguments[1..]),
        _ => Err(usage()),
    }
}

fn run_frontend(arguments: &[OsString]) -> Result<(), String> {
    let [project_path] = arguments else {
        return Err("usage: dm-conformance frontend <world.dme>".to_owned());
    };
    let project_path = PathBuf::from(project_path);
    let started = Instant::now();
    let compilation = dm_compiler::CompilerDatabase::new()
        .compile(&project_path)
        .map_err(|error| format!("failed to compile {}: {error}", project_path.display()))?;
    let stats = compilation.stats();
    println!("project_files={}", stats.project_files);
    println!("parsed_files={}", stats.parsed_files);
    println!("project_bytes={}", stats.project_bytes);
    println!("definitions={}", stats.definitions);
    println!("code_nodes={}", stats.code_nodes);
    println!("code_declarations={}", stats.code_declarations);
    println!("diagnostic_notes={}", stats.notes);
    println!("diagnostic_warnings={}", stats.warnings);
    println!("diagnostic_errors={}", stats.errors);
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
            "diagnostic={:?} {:?} {}: {}",
            diagnostic.severity, diagnostic.kind, location, diagnostic.message
        );
    }
    println!("elapsed_ms={}", started.elapsed().as_millis());
    Ok(())
}

fn run_project_compile_check(arguments: &[OsString]) -> Result<(), String> {
    let [project_path] = arguments else {
        return Err("usage: dm-conformance compile-check <world.dme>".to_owned());
    };
    let project_path = PathBuf::from(project_path);
    let started = Instant::now();
    let project = dm_project::Project::load(&project_path)
        .map_err(|error| format!("failed to load {}: {error}", project_path.display()))?;
    let mut total = 0_usize;
    let mut compiled = 0_usize;
    let mut error_categories = BTreeMap::new();
    for file in &project.files {
        if !matches!(
            file.kind,
            dm_project::FileKind::Environment | dm_project::FileKind::Source
        ) {
            continue;
        }
        let source = file
            .text()
            .map_err(|error| format!("{} is not UTF-8: {error}", file.path.display()))?;
        let syntax = dm_syntax::parse(source)
            .map_err(|error| format!("failed to parse {}: {error}", file.path.display()))?;
        for definition in syntax.definitions.iter().filter(|definition| {
            matches!(
                definition.kind,
                dm_syntax::DefinitionKind::Procedure
                    | dm_syntax::DefinitionKind::ProcedureOverride
                    | dm_syntax::DefinitionKind::Verb
            )
        }) {
            total += 1;
            match dm_vm::compile_procedure(definition) {
                Ok(_) => compiled += 1,
                Err(error) => {
                    *error_categories
                        .entry(compile_error_category(&error.message))
                        .or_insert(0_usize) += 1;
                }
            }
        }
    }
    let mut ranked_errors: Vec<_> = error_categories.into_iter().collect();
    ranked_errors.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    println!("procedures_total={total}");
    println!("procedures_compiled={compiled}");
    println!("procedures_unsupported={}", total - compiled);
    for (rank, (category, count)) in ranked_errors.into_iter().take(20).enumerate() {
        println!("unsupported_{}={} {:?}", rank + 1, count, category);
    }
    println!("elapsed_ms={}", started.elapsed().as_millis());
    Ok(())
}

fn compile_error_category(message: &str) -> String {
    for prefix in [
        "unknown local",
        "unexpected token",
        "unsupported binary operator",
        "unsupported unary operator",
        "expected an expression",
        "unexpected indentation",
    ] {
        if message.starts_with(prefix) {
            return prefix.to_owned();
        }
    }
    message.to_owned()
}

fn run_procedure(arguments: &[OsString]) -> Result<(), String> {
    let [source_path, procedure_path, raw_arguments @ ..] = arguments else {
        return Err(
            "usage: dm-conformance execute <source.dm> <procedure-path> [numeric-argument ...]"
                .to_owned(),
        );
    };
    let source_path = PathBuf::from(source_path);
    let procedure_path = procedure_path
        .to_str()
        .ok_or_else(|| "procedure path is not valid Unicode".to_owned())?;
    let source = fs::read_to_string(&source_path)
        .map_err(|error| format!("failed to read {}: {error}", source_path.display()))?;
    let syntax = dm_syntax::parse(&source)
        .map_err(|error| format!("failed to parse {}: {error}", source_path.display()))?;
    let definitions: Vec<_> = syntax
        .definitions
        .into_iter()
        .filter(|definition| {
            matches!(
                definition.kind,
                dm_syntax::DefinitionKind::Procedure
                    | dm_syntax::DefinitionKind::ProcedureOverride
                    | dm_syntax::DefinitionKind::Verb
            )
        })
        .collect();
    let module = dm_vm::compile_module(&definitions)
        .map_err(|error| format!("failed to compile {}: {error}", source_path.display()))?;
    let entry = module
        .procedure_id(procedure_path)
        .ok_or_else(|| format!("procedure {procedure_path} was not found"))?;
    let program = module
        .procedure(entry)
        .expect("a module resolves only valid procedure identities");
    let values: Result<Vec<_>, _> = raw_arguments
        .iter()
        .map(|argument| {
            let spelling = argument
                .to_str()
                .ok_or_else(|| "numeric argument is not valid Unicode".to_owned())?;
            spelling
                .parse::<f32>()
                .map(dm_vm::Value::number)
                .map_err(|error| format!("invalid numeric argument {spelling:?}: {error}"))
        })
        .collect();
    let result = dm_vm::execute_module(&module, entry, &values?)
        .map_err(|error| format!("runtime error in {procedure_path}: {error}"))?;

    println!("procedure={procedure_path}");
    println!("instructions={}", program.instructions.len());
    println!("locals={}", program.local_count);
    println!("result={result}");
    if let dm_vm::Value::Number(number) = result {
        println!("result_bits=0x{:08x}", number.bits());
    }
    Ok(())
}

fn run_project_syntax_check(arguments: &[OsString]) -> Result<(), String> {
    let [project_path] = arguments else {
        return Err("usage: dm-conformance check <world.dme>".to_owned());
    };
    let project_path = PathBuf::from(project_path);
    let started = Instant::now();
    let project = dm_project::Project::load(&project_path)
        .map_err(|error| format!("failed to load {}: {error}", project_path.display()))?;
    let mut checked_files = 0_usize;
    let mut definitions = 0_usize;
    let mut procedures = 0_usize;
    let mut variables = 0_usize;
    let mut types = 0_usize;
    for file in &project.files {
        if !matches!(
            file.kind,
            dm_project::FileKind::Environment | dm_project::FileKind::Source
        ) {
            continue;
        }
        let source = file
            .text()
            .map_err(|error| format!("{} is not UTF-8: {error}", file.path.display()))?;
        let syntax = dm_syntax::parse(source)
            .map_err(|error| format!("failed to parse {}: {error}", file.path.display()))?;
        checked_files += 1;
        definitions += syntax.definitions.len();
        for definition in syntax.definitions {
            match definition.kind {
                dm_syntax::DefinitionKind::Type => types += 1,
                dm_syntax::DefinitionKind::Procedure
                | dm_syntax::DefinitionKind::ProcedureOverride
                | dm_syntax::DefinitionKind::Verb => procedures += 1,
                dm_syntax::DefinitionKind::Variable
                | dm_syntax::DefinitionKind::VariableOverride => variables += 1,
            }
        }
    }
    println!("files_checked={checked_files}");
    println!("definitions={definitions}");
    println!("types={types}");
    println!("procedures={procedures}");
    println!("variables={variables}");
    println!("elapsed_ms={}", started.elapsed().as_millis());
    Ok(())
}

fn run_lexer(arguments: &[OsString]) -> Result<(), String> {
    let [source_path] = arguments else {
        return Err("usage: dm-conformance lex <source.dm>".to_owned());
    };
    let source_path = PathBuf::from(source_path);
    let source = fs::read_to_string(&source_path)
        .map_err(|error| format!("failed to read {}: {error}", source_path.display()))?;
    let tokens = dm_lexer::lex(&source).map_err(|error| {
        format!(
            "{}:{}..{}: {}",
            source_path.display(),
            error.span.start,
            error.span.end,
            error.message
        )
    })?;
    for token in tokens {
        println!("{}..{} {:?}", token.span.start, token.span.end, token.kind);
    }
    Ok(())
}

fn run_syntax(arguments: &[OsString]) -> Result<(), String> {
    let [source_path] = arguments else {
        return Err("usage: dm-conformance syntax <source.dm>".to_owned());
    };
    let source_path = PathBuf::from(source_path);
    let source = fs::read_to_string(&source_path)
        .map_err(|error| format!("failed to read {}: {error}", source_path.display()))?;
    let syntax = dm_syntax::parse(&source)
        .map_err(|error| format!("failed to parse {}: {error}", source_path.display()))?;
    for (index, definition) in syntax.definitions.iter().enumerate() {
        println!(
            "{index} path={} kind={:?} parent={:?} parameters={} body_lines={} span={}..{}",
            definition.path,
            definition.kind,
            definition.parent,
            definition.parameters.len(),
            definition.body.len(),
            definition.span.start,
            definition.span.end,
        );
    }
    println!("definitions={}", syntax.definitions.len());
    Ok(())
}

fn run_project(arguments: &[OsString]) -> Result<(), String> {
    let [project_path] = arguments else {
        return Err("usage: dm-conformance project <world.dme>".to_owned());
    };
    let project_path = PathBuf::from(project_path);
    let started = Instant::now();
    let project = dm_project::Project::load(&project_path)
        .map_err(|error| format!("failed to load {}: {error}", project_path.display()))?;

    let mut environments = 0;
    let mut sources = 0;
    let mut maps = 0;
    let mut interfaces = 0;
    let mut resources = 0;
    let mut bytes = 0_u64;
    for file in &project.files {
        match file.kind {
            dm_project::FileKind::Environment => environments += 1,
            dm_project::FileKind::Source => sources += 1,
            dm_project::FileKind::Map => maps += 1,
            dm_project::FileKind::Interface => interfaces += 1,
            dm_project::FileKind::Resource => resources += 1,
        }
        bytes += file.contents.len() as u64;
    }
    let system_includes = project
        .includes
        .iter()
        .filter(|include| matches!(include.target, dm_project::IncludeTarget::System(_)))
        .count();

    println!("root={}", project.root_directory.display());
    println!("files={}", project.files.len());
    println!("environment_files={environments}");
    println!("source_files={sources}");
    println!("map_files={maps}");
    println!("interface_files={interfaces}");
    println!("resource_files={resources}");
    println!("include_edges={}", project.includes.len());
    println!("system_includes={system_includes}");
    println!("bytes={bytes}");
    println!("elapsed_ms={}", started.elapsed().as_millis());
    Ok(())
}

fn run_compiler_probe(arguments: &[OsString]) -> Result<(), String> {
    let mut compiler = None;
    let mut project = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].to_str() {
            Some("--compiler") => {
                index += 1;
                compiler = arguments.get(index).map(PathBuf::from);
            }
            Some("--project") => {
                index += 1;
                project = arguments.get(index).map(PathBuf::from);
            }
            _ => return Err(probe_usage()),
        }
        index += 1;
    }
    let compiler = compiler.ok_or_else(probe_usage)?;
    let project = project.ok_or_else(probe_usage)?;
    probe_compiler(&compiler, &project, DEFAULT_TIMEOUT)
}

fn probe_compiler(compiler: &Path, project: &Path, timeout: Duration) -> Result<(), String> {
    let source_directory = project
        .parent()
        .ok_or_else(|| format!("project has no parent directory: {}", project.display()))?;
    let project_name = project
        .file_name()
        .ok_or_else(|| format!("project has no filename: {}", project.display()))?;
    let scratch = ScratchDirectory::new()?;
    copy_directory(source_directory, scratch.path())?;
    let mut child = Command::new(compiler)
        .arg(project_name)
        .current_dir(scratch.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start {}: {error}", compiler.display()))?;
    let stdout_reader = read_stream(
        child
            .stdout
            .take()
            .ok_or_else(|| "compiler stdout was not captured".to_owned())?,
    );
    let stderr_reader = read_stream(
        child
            .stderr
            .take()
            .ok_or_else(|| "compiler stderr was not captured".to_owned())?,
    );
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("failed to poll compiler: {error}"))?
        {
            let (stdout, stderr) = collect_streams(stdout_reader, stderr_reader)?;
            println!("outcome=exited");
            println!("status={status}");
            print_stream("stdout", &stdout);
            print_stream("stderr", &stderr);
            return Ok(());
        }
        if Instant::now() >= deadline {
            stop_child(&mut child)?;
            let (stdout, stderr) = collect_streams(stdout_reader, stderr_reader)?;
            println!("outcome=timeout");
            print_stream("stdout", &stdout);
            print_stream("stderr", &stderr);
            return Ok(());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn read_stream(mut stream: impl Read + Send + 'static) -> JoinHandle<io::Result<Vec<u8>>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn collect_streams(
    stdout_reader: JoinHandle<io::Result<Vec<u8>>>,
    stderr_reader: JoinHandle<io::Result<Vec<u8>>>,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let stdout = stdout_reader
        .join()
        .map_err(|_| "compiler stdout reader panicked".to_owned())?
        .map_err(|error| format!("failed to read compiler stdout: {error}"))?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "compiler stderr reader panicked".to_owned())?
        .map_err(|error| format!("failed to read compiler stderr: {error}"))?;
    Ok((stdout, stderr))
}

fn stop_child(child: &mut Child) -> Result<(), String> {
    child
        .kill()
        .map_err(|error| format!("compiler timed out and could not be stopped: {error}"))?;
    child
        .wait()
        .map_err(|error| format!("failed to reap timed-out compiler: {error}"))?;
    Ok(())
}

fn copy_directory(source: &Path, destination: &Path) -> Result<(), String> {
    for entry in fs::read_dir(source)
        .map_err(|error| format!("failed to read {}: {error}", source.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to read directory entry: {error}"))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .map_err(|error| format!("failed to inspect {}: {error}", source_path.display()))?;
        if file_type.is_dir() {
            fs::create_dir(&destination_path).map_err(|error| {
                format!("failed to create {}: {error}", destination_path.display())
            })?;
            copy_directory(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path).map_err(|error| {
                format!(
                    "failed to copy {} to {}: {error}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

struct ScratchDirectory {
    path: PathBuf,
}

impl ScratchDirectory {
    fn new() -> Result<Self, String> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock is before the Unix epoch: {error}"))?
            .as_nanos();
        let path = env::temp_dir().join(format!("dm64-conformance-{}-{unique}", process::id()));
        fs::create_dir(&path)
            .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!(
                "warning: failed to remove scratch directory {}: {error}",
                self.path.display()
            );
        }
    }
}

fn print_stream(name: &str, bytes: &[u8]) {
    let text = String::from_utf8_lossy(bytes);
    println!("{name}_bytes={}", bytes.len());
    if !text.is_empty() {
        println!("{name}_begin");
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
        println!("{name}_end");
    }
}

fn usage() -> String {
    "usage: dm-conformance <check|compile-check|execute|frontend|lex|probe|project|syntax> ..."
        .to_owned()
}

fn probe_usage() -> String {
    "usage: dm-conformance probe --compiler <dm.exe> --project <world.dme>".to_owned()
}
