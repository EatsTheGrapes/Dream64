use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use dm_compiler::persistent_database::{PersistentCompilerDatabase, encode_stable_id_table};
use dm_compiler::{
    BuildMode, CompilerDatabase, PERSISTENT_EXECUTABLE_PAGE_BASE, PERSISTENT_EXECUTABLE_SECTION,
};
use dm_lifecycle::artifact::{ArtifactSection, CompiledArtifact, engine_semantics_fingerprint};
use dm_lifecycle::{
    LifecycleIndex, build_parsed_dmm_cache, derive_lobby_readiness, dmm_measurements_from_parsed,
    encode_dmm_measurements, encode_parsed_dmm_cache, encode_procedure_semantics,
};
use dm_project::{Project, ProjectDefines};
use dm_runtime::{RuntimeImage, RuntimeStructuralSeed};
use dm_semantics::{ExecutableProcedures, ProcedureRegistry};

const BOOTSTRAP_SECTION: u32 = 1;
const EXECUTABLE_SECTION: u32 = 2;
const STRUCTURAL_SECTION: u32 = 3;
const COMPACT_WORDCODE_SECTION: u32 = 4;
const STABLE_ID_SECTION: u32 = 5;
const RUNTIME_LINKED_SECTION: u32 = 6;
const LIFECYCLE_DIRECTORY_SECTION: u32 = 7;
const DEFAULT_MAP_SECTION: u32 = 8;
const BOOT_MANIFEST_SECTION: u32 = 9;
const DMM_MEASUREMENT_SECTION: u32 = 10;
const PARSED_DMM_SECTION: u32 = 11;
const PROCEDURE_SEMANTICS_SECTION: u32 = 12;
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
    let mut arguments = env::args_os().skip(1).peekable();
    if arguments
        .peek()
        .is_some_and(|value| value == OsStr::new("compile"))
    {
        arguments.next();
    }
    let mut environment = None;
    let mut output = None;
    let mut reuse_artifact = None;
    let mut defines = ProjectDefines::new();
    while let Some(argument) = arguments.next() {
        if argument == OsStr::new("--output") && output.is_none() {
            let Some(path) = arguments.next() else {
                eprintln!("--output requires a path");
                return ExitCode::from(2);
            };
            output = Some(PathBuf::from(path));
        } else if argument == OsStr::new("--reuse-artifact") && reuse_artifact.is_none() {
            let Some(path) = arguments.next() else {
                eprintln!("--reuse-artifact requires a path");
                return ExitCode::from(2);
            };
            reuse_artifact = Some(PathBuf::from(path));
        } else if let Some(spec) = define_spec(&argument, &mut arguments) {
            let Some(spec) = spec else {
                eprintln!("-D/--define requires a NAME[=VALUE] argument");
                return ExitCode::from(2);
            };
            if let Err(error) = defines.push_spec(&spec) {
                eprintln!("invalid define {spec:?}: {error}");
                return ExitCode::from(2);
            }
        } else if !argument.to_string_lossy().starts_with('-') && environment.is_none() {
            environment = Some(PathBuf::from(argument));
        } else {
            print_usage();
            return ExitCode::from(2);
        }
    }
    let Some(environment) = environment else {
        print_usage();
        return ExitCode::from(2);
    };
    match compile(
        &environment,
        output.as_deref(),
        reuse_artifact.as_deref(),
        &defines,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", environment.display());
            ExitCode::FAILURE
        }
    }
}

/// Recognizes `-DNAME`, `-D NAME`, `--define NAME`, and `--define=NAME`.
///
/// Returns `None` when `argument` is not a define flag, `Some(None)` when it is
/// a define flag with a missing value, and `Some(Some(spec))` otherwise.
fn define_spec(
    argument: &OsStr,
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
) -> Option<Option<String>> {
    let argument = argument.to_str()?;
    if argument == "-D" || argument == "--define" {
        return Some(
            arguments
                .next()
                .and_then(|value| value.to_str().map(str::to_owned)),
        );
    }
    if let Some(spec) = argument.strip_prefix("-D") {
        return Some((!spec.is_empty()).then(|| spec.to_owned()));
    }
    if let Some(spec) = argument.strip_prefix("--define=") {
        return Some(Some(spec.to_owned()));
    }
    None
}

fn print_usage() {
    eprintln!(
        "usage: dream64-compiler [compile] <world.dme> [-D NAME[=VALUE] | --define NAME[=VALUE]]... [--output <world.d64>] [--reuse-artifact <previous.d64>]"
    );
}

fn compile(
    environment: &Path,
    output: Option<&Path>,
    reuse_artifact: Option<&Path>,
    defines: &ProjectDefines,
) -> Result<(), String> {
    let started = Instant::now();
    let cache_file = project_cache_file(environment);
    let compiler_database_file = compiler_database_file(environment);
    let artifact_file = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| environment.with_extension("d64"));
    let strict_source_hash = strict_source_hash_enabled();
    eprintln!(
        "compile-progress: source-validation-start mode={} project={}",
        if strict_source_hash {
            "strict-bytes"
        } else {
            "metadata"
        },
        environment.display()
    );
    let (project, project_hit, fingerprint) = if strict_source_hash {
        let (project, hit, fingerprint) = Project::load_cached_exact_with_fingerprint_and_defines(
            environment,
            &cache_file,
            defines,
        )
        .map_err(|error| format!("project snapshot: {error}"))?;
        (project, hit, *fingerprint.as_bytes())
    } else {
        let (project, hit) = Project::load_cached_with_defines(environment, &cache_file, defines)
            .map_err(|error| format!("project snapshot: {error}"))?;
        let fingerprint = *project.content_fingerprint().as_bytes();
        (project, hit, fingerprint)
    };
    eprintln!(
        "compile-progress: source-validation cache={} elapsed_ms={}",
        if project_hit { "hit" } else { "miss" },
        started.elapsed().as_millis(),
    );
    if project_hit
        && CompiledArtifact::read_from(&artifact_file, fingerprint)
            .is_ok_and(|artifact| artifact_has_current_sections(&artifact))
    {
        eprintln!(
            "compile-progress: artifact-ready cache=hit artifact={} elapsed_ms={}",
            artifact_file.display(),
            started.elapsed().as_millis(),
        );
        return Ok(());
    }
    if project_hit
        && reuse_artifact.is_some_and(|candidate| {
            candidate != artifact_file
                && CompiledArtifact::read_from(candidate, fingerprint)
                    .is_ok_and(|artifact| artifact_has_current_sections(&artifact))
        })
    {
        let candidate = reuse_artifact.expect("reuse artifact checked above");
        if let Some(parent) = artifact_file.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create artifact directory: {error}"))?;
        }
        fs::hard_link(candidate, &artifact_file)
            .or_else(|_| fs::copy(candidate, &artifact_file).map(|_| ()))
            .map_err(|error| format!("reuse prior runtime artifact: {error}"))?;
        eprintln!(
            "compile-progress: artifact-ready cache=reused artifact={} source={} elapsed_ms={}",
            artifact_file.display(),
            candidate.display(),
            started.elapsed().as_millis(),
        );
        return Ok(());
    }
    let phase = Instant::now();
    let (compilation, cache_stats) = CompilerDatabase::new()
        .compile_persistent_prevalidated(
            project,
            project_hit,
            &compiler_database_file,
            BuildMode::Incremental,
            persistent_semantic_digest(),
            persistent_build_configuration_digest(defines),
        )
        .map_err(|error| error.to_string())?;
    eprintln!(
        "compile-progress: persistent-frontend elapsed_ms={} linked_cache={} project_snapshot_cache={} parsed_syntax_cache={} syntax_reused={} syntax_reparsed={} changed_inputs={} invalidated_sections={} database={}",
        phase.elapsed().as_millis(),
        if cache_stats.linked_sections_reused != 0 {
            "hit"
        } else {
            "miss"
        },
        if cache_stats.project_snapshot_hit {
            "hit"
        } else {
            "miss"
        },
        if cache_stats.parsed_syntax_hit {
            "hit"
        } else {
            "miss"
        },
        cache_stats.syntax_files_reused,
        cache_stats.syntax_files_reparsed,
        cache_stats.changed_inputs,
        cache_stats.invalidated_sections,
        compiler_database_file.display(),
    );
    let mut persistent_database = PersistentCompilerDatabase::read(&compiler_database_file)
        .map_err(|error| format!("read persistent compiler database: {error}"))?;
    let procedure_ids = stable_ids_for_namespace(&persistent_database, "procedure");
    let type_ids = stable_ids_for_namespace(&persistent_database, "type");
    validate_procedure_id_inventory(&compilation, &procedure_ids)?;

    let phase = Instant::now();
    let procedures = ProcedureRegistry::build_with_stable_ids(&compilation, &procedure_ids)
        .map_err(|error| format!("stable procedure linking: {error}"))?;
    let procedure_digest = procedures.persistent_semantic_digest(&compilation);
    let cached_payload = persistent_database
        .sections
        .iter()
        .find(|section| section.section_id == PERSISTENT_EXECUTABLE_SECTION)
        .filter(|section| section.content_digest == procedure_digest)
        .and_then(|_| persistent_database.section_payload(PERSISTENT_EXECUTABLE_SECTION));
    let (executable, executable_cache_hit) = cached_payload
        .as_deref()
        .and_then(|payload| ExecutableProcedures::decode_compiled_artifact(payload).ok())
        .map_or_else(
            || {
                procedures
                    .compile_vm_all_symbolic_deferred(&compilation)
                    .map_err(|error| format!("complete executable lowering: {error}"))?
                    .into_fully_eager_bounded(MAX_EAGER_DIAGNOSTICS)
                    .map(|executable| (executable, false))
                    .map_err(|error| format!("complete executable lowering: {error}"))
            },
            |executable| Ok((executable, true)),
        )?;
    if executable.module().deferred_procedure_count() != 0 {
        return Err("complete executable lowering retained deferred procedures".to_owned());
    }
    eprintln!(
        "compile-progress: executable-link cache={} elapsed_ms={} procedures={}",
        if executable_cache_hit { "hit" } else { "miss" },
        phase.elapsed().as_millis(),
        executable.module().procedure_count(),
    );
    if let Some(selector) =
        env::var_os("DREAM64_DUMP_PROCEDURE").and_then(|value| value.into_string().ok())
    {
        dump_procedures(&executable, &selector);
        if env::var_os("DREAM64_DUMP_PROCEDURE_ONLY").is_some() {
            return Ok(());
        }
    }
    if !executable_cache_hit {
        let payload = executable.encode_compiled_artifact()?;
        let pages = persistent_database
            .replace_paged_section(
                PERSISTENT_EXECUTABLE_SECTION,
                PERSISTENT_EXECUTABLE_PAGE_BASE,
                procedure_digest,
                &payload,
                vec![1],
            )
            .map_err(|error| format!("cache executable link: {error}"))?;
        persistent_database
            .write_atomic(&compiler_database_file)
            .map_err(|error| format!("write executable compiler database: {error}"))?;
        eprintln!(
            "compile-progress: executable-cache-write bytes={} pages={pages}",
            payload.len()
        );
    }
    write_artifact(
        &artifact_file,
        &compilation,
        &executable,
        &procedures,
        &persistent_database,
        &type_ids,
    )?;
    eprintln!(
        "compile-progress: artifact-ready cache=miss artifact={} elapsed_ms={}",
        artifact_file.display(),
        started.elapsed().as_millis(),
    );
    Ok(())
}

fn dump_procedures(executable: &ExecutableProcedures, selector: &str) {
    let pc_from = env::var("DREAM64_DUMP_PC_FROM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let pc_to = env::var("DREAM64_DUMP_PC_TO")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    for (index, path) in executable.module().procedure_paths().enumerate() {
        if !path.contains(selector) {
            continue;
        }
        let Some(procedure) = executable.module().procedure_id_at(index) else {
            continue;
        };
        let Some(program) = executable.module().procedure(procedure) else {
            continue;
        };
        let digest = executable
            .module()
            .compute_procedure_semantic_digest(procedure)
            .map(|digest| {
                digest
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            })
            .unwrap_or_else(|error| format!("error:{error}"));
        eprintln!(
            "procedure-dump: path={path} id={} parameters={} locals={} instructions={} sha256={digest}",
            procedure.index(),
            program.parameter_count,
            program.local_count,
            program.instructions.len(),
        );
        for (pc, instruction) in program.instructions.iter().enumerate() {
            if pc < pc_from || pc > pc_to {
                continue;
            }
            eprintln!(
                "procedure-dump: pc={pc} source={:?} opcode={instruction:?}",
                program.source_spans.get(pc),
            );
        }
    }
}

fn artifact_has_current_sections(artifact: &CompiledArtifact) -> bool {
    (8..=11).contains(&artifact.sections().len())
        && artifact.section(EXECUTABLE_SECTION).is_some()
        && artifact.section(STRUCTURAL_SECTION).is_some()
        && artifact.section(STABLE_ID_SECTION).is_some()
        && artifact.section(RUNTIME_LINKED_SECTION).is_some()
        && artifact.section(LIFECYCLE_DIRECTORY_SECTION).is_some()
        && artifact.section(DMM_MEASUREMENT_SECTION).is_some()
        && artifact.section(PARSED_DMM_SECTION).is_some()
        && artifact.section(PROCEDURE_SEMANTICS_SECTION).is_some()
        && artifact.sections().iter().all(|section| {
            matches!(
                section.id(),
                BOOTSTRAP_SECTION
                    | EXECUTABLE_SECTION
                    | STRUCTURAL_SECTION
                    | COMPACT_WORDCODE_SECTION
                    | STABLE_ID_SECTION
                    | RUNTIME_LINKED_SECTION
                    | LIFECYCLE_DIRECTORY_SECTION
                    | DEFAULT_MAP_SECTION
                    | BOOT_MANIFEST_SECTION
                    | DMM_MEASUREMENT_SECTION
                    | PARSED_DMM_SECTION
                    | PROCEDURE_SEMANTICS_SECTION
            )
        })
}

fn strict_source_hash_enabled() -> bool {
    env::var_os("DREAM64_STRICT_SOURCE_HASH")
        .is_some_and(|value| !matches!(value.to_string_lossy().trim(), "" | "0" | "false" | "no"))
}

fn write_artifact(
    path: &Path,
    compilation: &dm_compiler::Compilation,
    executable: &ExecutableProcedures,
    procedures: &ProcedureRegistry,
    persistent_database: &PersistentCompilerDatabase,
    type_ids: &BTreeMap<String, u64>,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create artifact directory: {error}"))?;
    }
    let executable_payload = executable.encode_compiled_artifact()?;
    let procedure_semantics = encode_procedure_semantics(executable.module())?;
    let compilation_payload = compilation.encode_compiled_artifact();
    for path in [
        "/datum/parsed_map/proc/_tgm_load",
        "/datum/parsed_map/_tgm_load",
    ] {
        if let Some(procedure) = executable.module().effective_procedure_id(path) {
            let digest = executable
                .module()
                .compute_procedure_semantic_digest(procedure)?;
            let hex = digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            eprintln!(
                "compile-progress: procedure-semantic path={path} id={} sha256={hex}",
                procedure.index(),
            );
            break;
        }
    }
    let structural = RuntimeStructuralSeed::build_with_stable_ids(compilation, type_ids)
        .map_err(|error| format!("runtime structural seed: {error}"))?
        .encode_compiled_artifact();
    let stable_ids = encode_stable_id_table(&persistent_database.stable_ids)
        .map_err(|error| format!("stable-ID runtime section: {error}"))?;
    let mut runtime = RuntimeImage::from_compilation(compilation)
        .map_err(|error| format!("linked runtime image: {error}"))?;
    let boot_manifest = derive_lobby_readiness(compilation, &runtime)
        .map(|probe| probe.encode_portable_manifest())
        .transpose()
        .map_err(|error| format!("boot manifest: {error}"))?;
    let parsed_dmm_cache = build_parsed_dmm_cache(compilation)?;
    let dmm_measurements = dmm_measurements_from_parsed(&parsed_dmm_cache);
    let dmm_measurement_payload = encode_dmm_measurements(&dmm_measurements)?;
    let parsed_dmm_payload = encode_parsed_dmm_cache(&parsed_dmm_cache)?;
    eprintln!(
        "compile-progress: dmm-measurements entries={} bytes={}",
        dmm_measurements.len(),
        dmm_measurement_payload.len(),
    );
    runtime
        .materialize_linked_artifact_initializers(MAX_EAGER_DIAGNOSTICS)
        .map_err(|error| format!("linked runtime initializer module: {error}"))?;
    let runtime_linked = runtime
        .encode_linked_artifact(executable.module())
        .map_err(|error| format!("linked runtime image: {error}"))?;
    let lifecycle_directory = LifecycleIndex::build_compile_only(compilation, procedures)
        .encode_portable()
        .map_err(|error| format!("lifecycle directory: {error}"))?;
    let default_map = compilation
        .project()
        .files
        .iter()
        .find(|file| {
            file.relative_path
                .file_name()
                .is_some_and(|name| name.eq_ignore_ascii_case(OsStr::new("CentCom.dmm")))
        })
        .map(|file| {
            let source = file.text().map_err(|error| error.to_string())?;
            let map = dm_map::parse(source).map_err(|error| error.to_string())?;
            let plan = dm_world::build_plan(&map, compilation);
            dm_world::encode_named_portable_plan(&file.relative_path.display().to_string(), &plan)
                .map_err(|error| error.to_string())
        })
        .transpose()?;
    let compact = env::var_os("DREAM64_DISABLE_COMPACT_WORDCODE")
        .is_none()
        .then(|| {
            let image = dm_vm::CompactWordcodeImage::build(executable.module())
                .map_err(|error| format!("compact wordcode: {error}"))?;
            let payload = image
                .encode()
                .map_err(|error| format!("compact wordcode: {error}"))?;
            eprintln!(
                "compile-progress: compact-wordcode bytes={} strings={} procedures={} words={} specialized={}",
                payload.len(), image.string_count(), image.procedure_count(), image.word_count(),
                image.specialized_word_count(),
            );
            Ok::<_, String>(payload)
        })
        .transpose()?;
    let fingerprint = *compilation.project().content_fingerprint().as_bytes();
    let mut sections = vec![
        ArtifactSection::new(BOOTSTRAP_SECTION, compilation_payload),
        ArtifactSection::new(EXECUTABLE_SECTION, executable_payload),
        ArtifactSection::new(STRUCTURAL_SECTION, structural),
    ];
    if let Some(compact) = compact {
        sections.push(ArtifactSection::new(COMPACT_WORDCODE_SECTION, compact));
    }
    sections.push(ArtifactSection::new(STABLE_ID_SECTION, stable_ids));
    sections.push(ArtifactSection::new(RUNTIME_LINKED_SECTION, runtime_linked));
    sections.push(ArtifactSection::new(
        LIFECYCLE_DIRECTORY_SECTION,
        lifecycle_directory,
    ));
    if let Some(payload) = default_map {
        sections.push(ArtifactSection::new(DEFAULT_MAP_SECTION, payload));
    }
    if let Some(payload) = boot_manifest {
        sections.push(ArtifactSection::new(BOOT_MANIFEST_SECTION, payload));
    }
    sections.push(ArtifactSection::new(
        DMM_MEASUREMENT_SECTION,
        dmm_measurement_payload,
    ));
    sections.push(ArtifactSection::new(PARSED_DMM_SECTION, parsed_dmm_payload));
    sections.push(ArtifactSection::new(
        PROCEDURE_SEMANTICS_SECTION,
        procedure_semantics,
    ));
    let artifact = CompiledArtifact::new(fingerprint, sections)
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

fn stable_ids_for_namespace(
    database: &PersistentCompilerDatabase,
    namespace: &str,
) -> BTreeMap<String, u64> {
    database
        .stable_ids
        .iter()
        .filter(|entry| entry.namespace == namespace)
        .map(|entry| (entry.name.clone(), entry.id))
        .collect()
}

fn validate_procedure_id_inventory(
    compilation: &dm_compiler::Compilation,
    procedure_ids: &BTreeMap<String, u64>,
) -> Result<(), String> {
    let expected = compilation
        .code_tree()
        .nodes()
        .iter()
        .filter(|node| {
            matches!(
                node.kind,
                dm_object_tree::NodeKind::Procedure | dm_object_tree::NodeKind::Verb
            )
        })
        .count();
    if procedure_ids.len() != expected {
        return Err(format!(
            "persistent procedure ID inventory contains {} entries but the frontend contains {expected}",
            procedure_ids.len()
        ));
    }
    Ok(())
}

fn persistent_semantic_digest() -> [u8; 32] {
    digest32(
        // Bump whenever parsing, linking, or VM lowering semantics change.
        // This revision makes compiler-owned type predicates bypass synthetic
        // static procedure frames, so v2 executable pages are not reusable.
        b"dream64-compiler-semantics-v3-direct-type-predicates",
        &engine_semantics_fingerprint(),
    )
}

fn persistent_build_configuration_digest(defines: &ProjectDefines) -> [u8; 32] {
    let compact = if env::var_os("DREAM64_DISABLE_COMPACT_WORDCODE").is_some() {
        b"compact=disabled".as_slice()
    } else {
        b"compact=enabled".as_slice()
    };
    let mut value = compact.to_vec();
    for (name, replacement) in defines.iter() {
        value.push(0);
        value.extend_from_slice(name.as_bytes());
        value.push(b'=');
        value.extend_from_slice(replacement.as_bytes());
    }
    digest32(b"dream64-compiler-build-v2", &value)
}

fn digest32(domain: &[u8], value: &[u8]) -> [u8; 32] {
    let mut left = md5::Context::new();
    left.consume(domain);
    left.consume([0]);
    left.consume(value);
    let mut right = md5::Context::new();
    right.consume(domain);
    right.consume([1]);
    right.consume(value);
    let mut digest = [0; 32];
    digest[..16].copy_from_slice(&left.compute().0);
    digest[16..].copy_from_slice(&right.compute().0);
    digest
}

fn project_cache_file(environment: &Path) -> PathBuf {
    let canonical = fs::canonicalize(environment).unwrap_or_else(|_| environment.to_path_buf());
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    cache_root().join(format!("project-{hash:016x}.bin"))
}

fn compiler_database_file(environment: &Path) -> PathBuf {
    let project = project_cache_file(environment);
    let name = project
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("project");
    project.with_file_name(format!("compiler-{name}.d64cdb"))
}

fn cache_root() -> PathBuf {
    env::var_os("DREAM64_CACHE_DIR").map_or_else(
        || {
            env::current_exe()
                .ok()
                .and_then(|path| path.parent()?.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("target"))
                .join("dream64-cache")
        },
        PathBuf::from,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dm_compiler::persistent_database::{Digest, InputDependency, SectionDependency};

    fn artifact_with_sections(section_ids: &[u32]) -> CompiledArtifact {
        CompiledArtifact::new(
            [7; 16],
            section_ids
                .iter()
                .map(|id| ArtifactSection::new(*id, vec![*id as u8]))
                .collect(),
        )
        .expect("test artifact should be structurally valid")
    }

    #[test]
    fn artifact_cache_rejects_legacy_artifact_missing_new_required_section() {
        let legacy = artifact_with_sections(&[
            EXECUTABLE_SECTION,
            STRUCTURAL_SECTION,
            STABLE_ID_SECTION,
            RUNTIME_LINKED_SECTION,
            LIFECYCLE_DIRECTORY_SECTION,
        ]);
        assert!(!artifact_has_current_sections(&legacy));

        let current = artifact_with_sections(&[
            EXECUTABLE_SECTION,
            STRUCTURAL_SECTION,
            STABLE_ID_SECTION,
            RUNTIME_LINKED_SECTION,
            LIFECYCLE_DIRECTORY_SECTION,
            DMM_MEASUREMENT_SECTION,
            PARSED_DMM_SECTION,
            PROCEDURE_SEMANTICS_SECTION,
        ]);
        assert!(artifact_has_current_sections(&current));
    }

    #[test]
    fn missing_procedure_namespace_is_rejected_before_linking() {
        let root =
            std::env::temp_dir().join(format!("dream64-compiler-inventory-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let environment = root.join("world.dme");
        let source = root.join("code.dm");
        fs::write(&environment, "#include \"code.dm\"\n").unwrap();
        fs::write(&source, "/proc/run()\n\treturn 1\n").unwrap();
        let compilation = CompilerDatabase::new().compile(&environment).unwrap();
        let type_only_database = PersistentCompilerDatabase {
            semantic_digest: Digest::default(),
            build_configuration_digest: Digest::default(),
            inputs: Vec::<InputDependency>::new(),
            stable_ids: vec![dm_compiler::persistent_database::StableIdEntry {
                namespace: "type".to_owned(),
                name: "/datum".to_owned(),
                id: 0,
            }],
            sections: Vec::<SectionDependency>::new(),
        };
        let procedure_ids = stable_ids_for_namespace(&type_only_database, "procedure");
        assert!(validate_procedure_id_inventory(&compilation, &procedure_ids).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
