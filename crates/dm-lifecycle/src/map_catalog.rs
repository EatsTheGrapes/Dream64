/// Artifact-time DMM discovery, measurement, and portable catalog codecs.
///
/// This module owns filesystem discovery, path normalization, measurement,
/// and the versioned, checksummed portable encodings for DMM measurement
/// and parsed-DMM catalogs. It is the map parsing/cache products boundary
/// described in docs/ARCHITECTURE.md and must remain independent of
/// lifecycle resolution, execution/scheduling, readiness, and the durable
/// artifact envelope.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use dm_compiler::Compilation;
use dm_vm::DmmMeasurement;
use serde::{Deserialize, Serialize};
const DMM_MEASUREMENT_MAGIC: &[u8; 8] = b"D64DMMC\0";
const DMM_MEASUREMENT_VERSION: u16 = 1;
const MAX_DMM_MEASUREMENT_BYTES: usize = 64 * 1024 * 1024;
const MAX_DMM_MEASUREMENT_ENTRIES: usize = 1_000_000;
const MAX_DMM_PATH_BYTES: usize = 4096;
const MAX_DMM_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DMM_TOTAL_SOURCE_BYTES: u64 = 1024 * 1024 * 1024;
const PARSED_DMM_MAGIC: &[u8; 8] = b"D64PDMM\0";
// Version 2 stores TGM grid Y as the top source row, matching reader.dm.
const PARSED_DMM_VERSION: u16 = 2;
const MAX_PARSED_DMM_BYTES: u64 = 128 * 1024 * 1024;

/// One content-addressed artifact-time DMM measurement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortableDmmMeasurement {
    /// MD5 content identity of the exact map bytes compiled into the artifact.
    pub digest: [u8; 16],
    /// Exact coordinate bounds in BYOND MAP_MINX..MAP_MAXZ order.
    pub measurement: DmmMeasurement,
}

/// One source-ordered coordinate/grid record from a parsed DMM resource.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortableDmmGrid {
    /// One-based starting X coordinate.
    pub x: i32,
    /// One-based starting Y coordinate.
    pub y: i32,
    /// One-based Z coordinate.
    pub z: i32,
    /// Grid lines in their original top-to-bottom order.
    pub lines: Vec<String>,
}

/// Complete parser product needed to materialize `/datum/parsed_map` fields.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortableParsedDmm {
    /// MD5 identity of the exact source bytes.
    pub digest: [u8; 16],
    /// `true` for TGM formatting and `false` for ordinary DMM formatting.
    pub tgm: bool,
    /// Uniform map-key byte width.
    pub key_len: u32,
    /// Byte length of the first grid line.
    pub line_len: u32,
    /// Bounds in BYOND MAP_MINX..MAP_MAXZ order.
    pub bounds: [i32; 6],
    /// Key and raw model-body pairs in source definition order.
    pub models: Vec<(String, String)>,
    /// Coordinate blocks in source order.
    pub grids: Vec<PortableDmmGrid>,
}

/// Builds measurements for every valid map resource discovered by the project loader.
pub fn build_dmm_measurements(
    compilation: &Compilation,
) -> Result<BTreeMap<String, PortableDmmMeasurement>, String> {
    Ok(dmm_measurements_from_parsed(&build_parsed_dmm_cache(
        compilation,
    )?))
}

/// Derives the compact section-10 catalog without reparsing map sources.
#[must_use]
pub fn dmm_measurements_from_parsed(
    parsed: &BTreeMap<String, PortableParsedDmm>,
) -> BTreeMap<String, PortableDmmMeasurement> {
    parsed
        .iter()
        .map(|(path, entry)| {
            (
                path.clone(),
                PortableDmmMeasurement {
                    digest: entry.digest,
                    measurement: DmmMeasurement {
                        digest: entry.digest,
                        bounds: entry.bounds,
                    },
                },
            )
        })
        .collect()
}

/// Builds the complete source-ordered parsed-map catalog in one bounded scan.
pub fn build_parsed_dmm_cache(
    compilation: &Compilation,
) -> Result<BTreeMap<String, PortableParsedDmm>, String> {
    let root = fs::canonicalize(&compilation.project().root_directory)
        .map_err(|error| format!("canonicalize DMM project root: {error}"))?;
    let mut paths = Vec::new();
    discover_dmm_paths(&root, &root, &mut paths)?;
    paths.sort_unstable();
    if paths.len() > MAX_DMM_MEASUREMENT_ENTRIES {
        return Err("project contains too many DMM resources".to_owned());
    }
    let mut total_bytes = 0u64;
    let mut parsed_cache = BTreeMap::new();
    for path in paths {
        let metadata = fs::metadata(&path)
            .map_err(|error| format!("inspect DMM resource {}: {error}", path.display()))?;
        if metadata.len() > MAX_DMM_SOURCE_BYTES {
            return Err(format!(
                "DMM resource {} exceeds its byte limit",
                path.display()
            ));
        }
        total_bytes = total_bytes
            .checked_add(metadata.len())
            .ok_or("DMM resource byte total overflow")?;
        if total_bytes > MAX_DMM_TOTAL_SOURCE_BYTES {
            return Err("project DMM resources exceed their aggregate byte limit".to_owned());
        }
        let relative = path
            .strip_prefix(&root)
            .map_err(|_| format!("DMM resource escapes project root: {}", path.display()))?;
        let portable = normalize_portable_dmm_path(&relative.to_string_lossy())
            .ok_or_else(|| format!("invalid project-relative DMM path: {}", relative.display()))?;
        let bytes = fs::read(&path)
            .map_err(|error| format!("read DMM resource {}: {error}", path.display()))?;
        let source = std::str::from_utf8(&bytes)
            .map_err(|_| format!("DMM resource is not UTF-8: {}", path.display()))?;
        let Ok(map) = dm_map::parse(source) else {
            // Invalid/non-map `.dmm` resources retain the runtime DM parser fallback.
            continue;
        };
        let Some(measurement) = measure_parsed_dmm(source, &map) else {
            continue;
        };
        let digest = md5::compute(&bytes).0;
        let mut definitions = map.keys.values().collect::<Vec<_>>();
        definitions.sort_unstable_by_key(|definition| definition.span.start);
        let mut models = Vec::with_capacity(definitions.len());
        let mut tgm = false;
        for (index, definition) in definitions.into_iter().enumerate() {
            let raw = source
                .get(definition.span.start..definition.span.end)
                .ok_or_else(|| format!("invalid DMM definition span in {portable:?}"))?;
            let equals = raw
                .find('=')
                .ok_or_else(|| format!("missing DMM model assignment in {portable:?}"))?;
            let open = raw[equals + 1..]
                .find('(')
                .map(|offset| equals + 1 + offset)
                .ok_or_else(|| format!("missing DMM model opener in {portable:?}"))?;
            let mut body = raw
                .get(open + 1..raw.len().saturating_sub(1))
                .ok_or_else(|| format!("invalid DMM model body in {portable:?}"))?;
            if body.starts_with('\n') {
                if index == 0 {
                    tgm = true;
                }
                body = &body[1..];
            }
            models.push((definition.key.clone(), body.to_owned()));
        }
        let grids = map
            .blocks
            .iter()
            .map(|block| PortableDmmGrid {
                x: block.x,
                // reader.dm stores TGM columns from top to bottom. After
                // splitting the block it advances ycrd by line_count - 1 so
                // `_tgm_load` can decrement Y for each subsequent key. DMM
                // blocks retain their original lower-left Y coordinate.
                y: if tgm {
                    block.y.saturating_add(
                        i32::try_from(block.rows.len().saturating_sub(1)).unwrap_or(i32::MAX),
                    )
                } else {
                    block.y
                },
                z: block.z,
                lines: block
                    .rows
                    .iter()
                    .map(|row| row.concat())
                    .collect::<Vec<_>>(),
            })
            .collect::<Vec<_>>();
        let line_len = grids
            .first()
            .and_then(|grid| grid.lines.first())
            .map_or(0, String::len);
        let key_len = u32::try_from(map.key_width)
            .map_err(|_| format!("DMM key width exceeds u32 in {portable:?}"))?;
        let line_len = u32::try_from(line_len)
            .map_err(|_| format!("DMM line length exceeds u32 in {portable:?}"))?;
        if parsed_cache
            .insert(
                portable.clone(),
                PortableParsedDmm {
                    digest,
                    tgm,
                    key_len,
                    line_len,
                    bounds: measurement.bounds,
                    models,
                    grids,
                },
            )
            .is_some()
        {
            return Err(format!(
                "duplicate normalized DMM resource path {portable:?}"
            ));
        }
    }
    Ok(parsed_cache)
}

fn discover_dmm_paths(
    root: &Path,
    directory: &Path,
    output: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| format!("scan DMM directory {}: {error}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("scan DMM directory {}: {error}", directory.display()))?;
    entries.sort_unstable_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("inspect project entry {}: {error}", path.display()))?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if matches!(name.as_ref(), ".git" | "target" | "node_modules" | ".cache") {
                continue;
            }
            let canonical = fs::canonicalize(&path).map_err(|error| {
                format!("canonicalize project directory {}: {error}", path.display())
            })?;
            if !canonical.starts_with(root) {
                return Err(format!(
                    "project directory escapes root: {}",
                    path.display()
                ));
            }
            discover_dmm_paths(root, &canonical, output)?;
        } else if file_type.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("dmm"))
        {
            output.push(path);
        }
    }
    Ok(())
}

/// Measures exact coordinate bounds using the shared loss-aware DMM parser.
#[must_use]
pub fn measure_dmm_source(source: &str) -> Option<DmmMeasurement> {
    let map = dm_map::parse(source).ok()?;
    measure_parsed_dmm(source, &map)
}

fn measure_parsed_dmm(source: &str, map: &dm_map::Map) -> Option<DmmMeasurement> {
    let first = map.blocks.first()?;
    let mut bounds = [first.x, first.y, first.z, first.x, first.y, first.z];
    for block in &map.blocks {
        let width = block.rows.first().map_or(0, Vec::len) as i32;
        let height = block.rows.len() as i32;
        bounds[0] = bounds[0].min(block.x);
        bounds[1] = bounds[1].min(block.y);
        bounds[2] = bounds[2].min(block.z);
        bounds[3] = bounds[3].max(block.x + width.saturating_sub(1));
        bounds[4] = bounds[4].max(block.y + height.saturating_sub(1));
        bounds[5] = bounds[5].max(block.z);
    }
    Some(DmmMeasurement {
        digest: md5::compute(source.as_bytes()).0,
        bounds,
    })
}

fn normalize_portable_dmm_path(path: &str) -> Option<String> {
    let path = path.replace('\\', "/");
    if path.is_empty() || path.starts_with('/') || path.contains(':') {
        return None;
    }
    let mut components = Vec::new();
    for component in path.split('/') {
        match component {
            "" | "." => {}
            ".." => return None,
            component => components.push(component.to_ascii_lowercase()),
        }
    }
    (!components.is_empty()).then(|| components.join("/"))
}

/// Encodes a bounded, versioned, checksummed DMM measurement catalog.
pub fn encode_dmm_measurements(
    measurements: &BTreeMap<String, PortableDmmMeasurement>,
) -> Result<Vec<u8>, String> {
    if measurements.len() > MAX_DMM_MEASUREMENT_ENTRIES {
        return Err("DMM measurement catalog has too many entries".to_owned());
    }
    let mut payload = Vec::new();
    payload.extend_from_slice(&(measurements.len() as u32).to_le_bytes());
    for (path, entry) in measurements {
        if path.len() > MAX_DMM_PATH_BYTES
            || normalize_portable_dmm_path(path).as_deref() != Some(path)
        {
            return Err(format!("invalid portable DMM path {path:?}"));
        }
        if entry.digest != entry.measurement.digest {
            return Err(format!("DMM measurement digest mismatch for {path:?}"));
        }
        payload.extend_from_slice(&(path.len() as u32).to_le_bytes());
        payload.extend_from_slice(path.as_bytes());
        payload.extend_from_slice(&entry.digest);
        for coordinate in entry.measurement.bounds {
            payload.extend_from_slice(&coordinate.to_le_bytes());
        }
    }
    if payload.len() > MAX_DMM_MEASUREMENT_BYTES {
        return Err("DMM measurement catalog exceeds its byte limit".to_owned());
    }
    let mut encoded = Vec::with_capacity(22 + payload.len());
    encoded.extend_from_slice(DMM_MEASUREMENT_MAGIC);
    encoded.extend_from_slice(&DMM_MEASUREMENT_VERSION.to_le_bytes());
    encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

/// Decodes and validates a portable DMM measurement catalog.
pub fn decode_dmm_measurements(
    bytes: &[u8],
) -> Result<BTreeMap<String, PortableDmmMeasurement>, String> {
    if bytes.len() < 22 || &bytes[..8] != DMM_MEASUREMENT_MAGIC {
        return Err("invalid DMM measurement catalog header".to_owned());
    }
    if u16::from_le_bytes([bytes[8], bytes[9]]) != DMM_MEASUREMENT_VERSION {
        return Err("unsupported DMM measurement catalog version".to_owned());
    }
    let length = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    if length > MAX_DMM_MEASUREMENT_BYTES as u64 || length as usize != bytes.len() - 22 {
        return Err("invalid DMM measurement catalog length".to_owned());
    }
    let checksum = u32::from_le_bytes(bytes[18..22].try_into().unwrap());
    let payload = &bytes[22..];
    if crc32fast::hash(payload) != checksum {
        return Err("DMM measurement catalog checksum mismatch".to_owned());
    }
    let mut cursor = 0usize;
    let take = |cursor: &mut usize, count: usize| -> Result<&[u8], String> {
        let end = cursor
            .checked_add(count)
            .ok_or("DMM catalog offset overflow")?;
        let value = payload
            .get(*cursor..end)
            .ok_or("truncated DMM measurement catalog")?;
        *cursor = end;
        Ok(value)
    };
    let count = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
    if count > MAX_DMM_MEASUREMENT_ENTRIES {
        return Err("DMM measurement catalog has too many entries".to_owned());
    }
    let mut decoded = BTreeMap::new();
    for _ in 0..count {
        let path_len = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
        if path_len > MAX_DMM_PATH_BYTES {
            return Err("DMM measurement path exceeds its limit".to_owned());
        }
        let path = std::str::from_utf8(take(&mut cursor, path_len)?)
            .map_err(|_| "DMM measurement path is not UTF-8")?
            .to_owned();
        if normalize_portable_dmm_path(&path).as_deref() != Some(path.as_str()) {
            return Err("DMM measurement path is not normalized".to_owned());
        }
        let digest = take(&mut cursor, 16)?.try_into().unwrap();
        let mut bounds = [0; 6];
        for coordinate in &mut bounds {
            *coordinate = i32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap());
        }
        if decoded
            .insert(
                path,
                PortableDmmMeasurement {
                    digest,
                    measurement: DmmMeasurement { digest, bounds },
                },
            )
            .is_some()
        {
            return Err("duplicate DMM measurement path".to_owned());
        }
    }
    if cursor != payload.len() {
        return Err("trailing DMM measurement catalog bytes".to_owned());
    }
    Ok(decoded)
}

/// Encodes the complete parsed DMM catalog as a bounded, checksummed payload.
pub fn encode_parsed_dmm_cache(
    parsed: &BTreeMap<String, PortableParsedDmm>,
) -> Result<Vec<u8>, String> {
    use bincode::Options as _;
    validate_parsed_dmm_cache(parsed)?;
    let options = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_PARSED_DMM_BYTES);
    let payload = options
        .serialize(parsed)
        .map_err(|error| format!("encode parsed DMM cache: {error}"))?;
    let mut encoded = Vec::with_capacity(22 + payload.len());
    encoded.extend_from_slice(PARSED_DMM_MAGIC);
    encoded.extend_from_slice(&PARSED_DMM_VERSION.to_le_bytes());
    encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

/// Decodes and fully validates a complete parsed DMM catalog.
pub fn decode_parsed_dmm_cache(
    bytes: &[u8],
) -> Result<BTreeMap<String, PortableParsedDmm>, String> {
    use bincode::Options as _;
    if bytes.len() < 22 || &bytes[..8] != PARSED_DMM_MAGIC {
        return Err("invalid parsed DMM cache header".to_owned());
    }
    if u16::from_le_bytes([bytes[8], bytes[9]]) != PARSED_DMM_VERSION {
        return Err("unsupported parsed DMM cache version".to_owned());
    }
    let length = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    if length > MAX_PARSED_DMM_BYTES || length as usize != bytes.len() - 22 {
        return Err("invalid parsed DMM cache length".to_owned());
    }
    let checksum = u32::from_le_bytes(bytes[18..22].try_into().unwrap());
    if crc32fast::hash(&bytes[22..]) != checksum {
        return Err("parsed DMM cache checksum mismatch".to_owned());
    }
    let options = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_PARSED_DMM_BYTES)
        .reject_trailing_bytes();
    let parsed = options
        .deserialize(&bytes[22..])
        .map_err(|error| format!("decode parsed DMM cache: {error}"))?;
    validate_parsed_dmm_cache(&parsed)?;
    Ok(parsed)
}

fn validate_parsed_dmm_cache(parsed: &BTreeMap<String, PortableParsedDmm>) -> Result<(), String> {
    if parsed.len() > MAX_DMM_MEASUREMENT_ENTRIES {
        return Err("parsed DMM cache has too many entries".to_owned());
    }
    let mut aggregate = 0u64;
    for (path, entry) in parsed {
        if path.len() > MAX_DMM_PATH_BYTES
            || normalize_portable_dmm_path(path).as_deref() != Some(path)
        {
            return Err(format!("invalid parsed DMM path {path:?}"));
        }
        if entry.key_len == 0 || entry.models.is_empty() || entry.grids.is_empty() {
            return Err(format!("incomplete parsed DMM entry {path:?}"));
        }
        for (key, model) in &entry.models {
            if key.len() != entry.key_len as usize {
                return Err(format!("inconsistent map key width in {path:?}"));
            }
            aggregate = aggregate
                .checked_add((key.len() + model.len()) as u64)
                .ok_or("parsed DMM cache size overflow")?;
        }
        for grid in &entry.grids {
            for line in &grid.lines {
                if line.len() != entry.line_len as usize || line.len() % entry.key_len as usize != 0
                {
                    return Err(format!("inconsistent grid line width in {path:?}"));
                }
                aggregate = aggregate
                    .checked_add(line.len() as u64)
                    .ok_or("parsed DMM cache size overflow")?;
            }
        }
        if aggregate > MAX_PARSED_DMM_BYTES {
            return Err("parsed DMM cache content exceeds its limit".to_owned());
        }
    }
    Ok(())
}
