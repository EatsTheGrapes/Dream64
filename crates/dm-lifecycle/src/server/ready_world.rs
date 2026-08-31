//! Content-addressed ready-world snapshot cache: the gzip-compressed VM state
//! image written once a headless boot reaches authoritative readiness and
//! restored on a later boot of the same engine/project/map (and, for
//! production, the same deployment identity).

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write as _};
use std::path::{Path, PathBuf};

use dm_compiler::Compilation;
use dm_lifecycle::artifact::engine_semantics_fingerprint;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

use super::cli::ProductionReadyWorldIdentity;

pub(crate) fn restore_ready_world_cache(
    path: &Path,
    state: &mut dm_vm::ExecutionState,
    module: &dm_vm::Module,
) -> Result<u64, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let bytes = file.metadata().map_err(|error| error.to_string())?.len();
    state
        .restore_ready_world_snapshot_from(&mut GzDecoder::new(BufReader::new(file)), module)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

pub(crate) fn write_ready_world_cache(
    path: &Path,
    state: &dm_vm::ExecutionState,
) -> Result<u64, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut temporary = None;
    for sequence in 0..100_u32 {
        let candidate =
            path.with_extension(format!("ready.tmp.{}.{}", std::process::id(), sequence));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    let (temporary_path, file) = temporary
        .ok_or_else(|| "could not reserve a ready-world cache temporary file".to_owned())?;
    let result = (|| {
        let mut writer = GzEncoder::new(BufWriter::new(file), Compression::fast());
        state
            .write_ready_world_snapshot_to(&mut writer)
            .map_err(|error| error.to_string())?;
        let mut writer = writer.finish().map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        let file = writer.into_inner().map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        let bytes = file.metadata().map_err(|error| error.to_string())?.len();
        drop(file);
        if path.exists() {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        }
        fs::rename(&temporary_path, path).map_err(|error| error.to_string())?;
        Ok(bytes)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

pub(crate) fn ready_world_cache_file(
    project_cache: &Path,
    map_source: &str,
    compilation: &Compilation,
    production_identity: Option<&ProductionReadyWorldIdentity>,
) -> PathBuf {
    let mut identity = md5::Context::new();
    identity.consume(b"dream64-ready-world-v1");
    identity.consume(engine_semantics_fingerprint());
    identity.consume(env!("DREAM64_ENGINE_TARGET").as_bytes());
    identity.consume(compilation.project().content_fingerprint().as_bytes());
    identity.consume(map_source.as_bytes());
    if let Some(production_identity) = production_identity {
        identity.consume(b"production-ready-world-v1");
        identity.consume(production_identity.random_seed.to_le_bytes());
        identity.consume(production_identity.deployment_id.as_bytes());
    }
    let digest = identity.compute();
    project_cache.with_file_name(format!("ready-{digest:x}.bin"))
}
