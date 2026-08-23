use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::CompilerDatabase;
use dm_lifecycle::artifact::{ArtifactSection, CompiledArtifact};
use dm_project::Project;
use dm_runtime::RuntimeStructuralSeed;
use dm_semantics::{ExecutableProcedures, ProcedureRegistry};

const BOOTSTRAP_SECTION: u32 = 1;
const EXECUTABLE_SECTION: u32 = 2;
const STRUCTURAL_SECTION: u32 = 3;
const MAX_EAGER_DIAGNOSTICS: usize = 32;

fn main() -> ExitCode {
    std::thread::Builder::new()
        .name("dream64-compiler".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .expect("failed to create Dream64 compiler thread")
        .join()
        .unwrap_or_else(|_| {
            eprintln!("Dream64 compiler thread panicked");
            ExitCode::FAILURE
        })
}

fn run() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    let first = arguments.next();
    let environment = match first.as_deref() {
        Some(value) if value == OsStr::new("compile") => arguments.next(),
        value => value.map(Into::into),
    };
    let Some(environment) = environment.map(PathBuf::from) else {
        eprintln!("usage: dream64-compiler [compile] <world.dme>");
        return ExitCode::from(2);
    };
    if arguments.next().is_some() {
        eprintln!("usage: dream64-compiler [compile] <world.dme>");
        return ExitCode::from(2);
    }
    match compile(&environment) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", environment.display());
            ExitCode::FAILURE
        }
    }
}

fn compile(environment: &Path) -> Result<(), String> {
    let started = Instant::now();
    let cache_file = project_cache_file(environment);
    let artifact_file = environment.with_extension("d64");
    let (project, project_hit) = Project::load_cached(environment, &cache_file)
        .map_err(|error| format!("project snapshot: {error}"))?;
    let fingerprint = *project.content_fingerprint().as_bytes();
    eprintln!(
        "compile-progress: source-validation cache={} elapsed_ms={}",
        if project_hit { "hit" } else { "miss" },
        started.elapsed().as_millis(),
    );
    if project_hit
        && CompiledArtifact::read_from(&artifact_file, fingerprint).is_ok_and(|artifact| {
            artifact.sections().len() == 3
                && artifact.section(BOOTSTRAP_SECTION).is_some()
                && artifact.section(EXECUTABLE_SECTION).is_some()
                && artifact.section(STRUCTURAL_SECTION).is_some()
        })
    {
        eprintln!(
            "compile-progress: artifact-ready cache=hit artifact={} elapsed_ms={}",
            artifact_file.display(),
            started.elapsed().as_millis(),
        );
        return Ok(());
    }
    drop(project);

    let phase = Instant::now();
    let (compilation, cache_stats) = CompilerDatabase::new()
        .compile_cached_with_stats(environment, &cache_file)
        .map_err(|error| error.to_string())?;
    eprintln!(
        "compile-progress: frontend elapsed_ms={} parsed_syntax_cache={} syntax_reused={} syntax_reparsed={}",
        phase.elapsed().as_millis(),
        if cache_stats.parsed_syntax_hit {
            "hit"
        } else {
            "miss"
        },
        cache_stats.syntax_files_reused,
        cache_stats.syntax_files_reparsed,
    );
    let phase = Instant::now();
    let procedures = ProcedureRegistry::build(&compilation);
    let executable = procedures
        .compile_vm_all_symbolic_deferred(&compilation)
        .map_err(|error| format!("complete executable lowering: {error}"))?
        .into_fully_eager_bounded(MAX_EAGER_DIAGNOSTICS)
        .map_err(|error| format!("complete executable lowering: {error}"))?;
    if executable.module().deferred_procedure_count() != 0 {
        return Err("complete executable lowering retained deferred procedures".to_owned());
    }
    eprintln!(
        "compile-progress: lowering elapsed_ms={} procedures={}",
        phase.elapsed().as_millis(),
        executable.module().procedure_count(),
    );
    write_artifact(&artifact_file, &compilation, &executable)?;
    eprintln!(
        "compile-progress: artifact-ready cache=miss artifact={} elapsed_ms={}",
        artifact_file.display(),
        started.elapsed().as_millis(),
    );
    Ok(())
}

fn write_artifact(
    path: &Path,
    compilation: &dm_compiler::Compilation,
    executable: &ExecutableProcedures,
) -> Result<(), String> {
    let bootstrap = compilation.encode_compiled_artifact();
    let executable = executable.encode_compiled_artifact()?;
    let structural = RuntimeStructuralSeed::build(compilation)
        .map_err(|error| format!("runtime structural seed: {error}"))?
        .encode_compiled_artifact();
    let fingerprint = *compilation.project().content_fingerprint().as_bytes();
    let artifact = CompiledArtifact::new(
        fingerprint,
        vec![
            ArtifactSection::new(BOOTSTRAP_SECTION, bootstrap),
            ArtifactSection::new(EXECUTABLE_SECTION, executable),
            ArtifactSection::new(STRUCTURAL_SECTION, structural),
        ],
    )
    .map_err(|error| format!("build runtime artifact: {error}"))?;
    let stats = artifact
        .write_atomic_with_stats(path)
        .map_err(|error| format!("write runtime artifact: {error}"))?;
    eprintln!(
        "compile-progress: artifact-write bytes={} payload_bytes={} peak_staging_bytes={} write_calls={}",
        stats.encoded_bytes, stats.payload_bytes, stats.peak_staging_bytes, stats.write_calls,
    );
    Ok(())
}

fn project_cache_file(environment: &Path) -> PathBuf {
    let canonical = fs::canonicalize(environment).unwrap_or_else(|_| environment.to_path_buf());
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let cache_root = env::var_os("DREAM64_CACHE_DIR").map_or_else(
        || {
            env::current_exe()
                .ok()
                .and_then(|path| path.parent()?.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("target"))
                .join("dream64-cache")
        },
        PathBuf::from,
    );
    cache_root.join(format!("project-{hash:016x}.bin"))
}
