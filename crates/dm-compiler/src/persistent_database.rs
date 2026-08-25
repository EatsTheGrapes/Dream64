//! Versioned persistent state for incremental compilation.
//!
//! This format is intentionally independent from the legacy project and
//! parsed-syntax caches. It records identities and link metadata needed to
//! decide which future compiler stages may be reused.

use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"D64CDB\0\0";
const VERSION: u32 = 1;
const STABLE_ID_MAGIC: &[u8; 8] = b"D64SID\0\0";
const STABLE_ID_VERSION: u32 = 1;
const HEADER_LEN: u64 = 8 + 4 + 4 + 8 + 16;
const MAX_DATABASE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Maximum independently reusable payload page. Aggregate databases may be larger.
pub const MAX_SECTION_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
const MAX_RECORDS: u64 = 1_000_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// Opaque content or configuration digest stored in the compiler database.
pub type Digest = [u8; 32];

/// One normalized project input and its exact content identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputDependency {
    /// Slash-separated, project-relative identity.
    pub identity: String,
    /// Exact content digest supplied by the caller.
    pub content_digest: Digest,
    /// Input length used as a cheap validation prefilter.
    pub byte_length: u64,
}

impl InputDependency {
    /// Creates an input record with a normalized lexical identity.
    #[must_use]
    pub fn new(path: impl AsRef<Path>, content_digest: Digest, byte_length: u64) -> Self {
        Self {
            identity: normalize_identity(path.as_ref()),
            content_digest,
            byte_length,
        }
    }
}

/// A stable numeric identity assigned by the linker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StableIdEntry {
    /// Identity domain, such as `type`, `proc`, `field`, or `resource`.
    pub namespace: String,
    /// Canonical name within the domain.
    pub name: String,
    /// Stable native-width-independent 64-bit ID.
    pub id: u64,
}

/// Encodes the stable linker identities for embedding in a runtime artifact.
///
/// The table has its own magic, version, checksum, and bounded 64-bit record
/// count so runtime artifacts never need to expose or decode compiler cache
/// internals.
pub fn encode_stable_id_table(entries: &[StableIdEntry]) -> Result<Vec<u8>, DatabaseError> {
    validate_stable_ids(entries)?;
    let mut payload = Vec::new();
    write_u64(&mut payload, entries.len() as u64);
    for entry in entries {
        write_string(&mut payload, &entry.namespace)?;
        write_string(&mut payload, &entry.name)?;
        write_u64(&mut payload, entry.id);
    }
    if payload.len() as u64 > MAX_DATABASE_BYTES - HEADER_LEN {
        return Err(DatabaseError::Limit("stable-ID payload"));
    }
    let checksum = md5::compute(&payload);
    let mut output = Vec::with_capacity(HEADER_LEN as usize + payload.len());
    output.extend_from_slice(STABLE_ID_MAGIC);
    output.extend_from_slice(&STABLE_ID_VERSION.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    write_u64(&mut output, payload.len() as u64);
    output.extend_from_slice(&checksum.0);
    output.extend_from_slice(&payload);
    Ok(output)
}

/// Decodes and validates an artifact-embedded stable linker identity table.
pub fn decode_stable_id_table(bytes: &[u8]) -> Result<Vec<StableIdEntry>, DatabaseError> {
    if bytes.len() as u64 > MAX_DATABASE_BYTES || (bytes.len() as u64) < HEADER_LEN {
        return Err(DatabaseError::Format("stable-ID table length"));
    }
    let mut input = Cursor::new(bytes);
    let mut magic = [0; 8];
    input.read_exact(&mut magic)?;
    if &magic != STABLE_ID_MAGIC {
        return Err(DatabaseError::Format("stable-ID magic"));
    }
    if read_u32(&mut input)? != STABLE_ID_VERSION || read_u32(&mut input)? != 0 {
        return Err(DatabaseError::Format("stable-ID version or flags"));
    }
    let payload_len = read_u64(&mut input)?;
    let mut expected = [0; 16];
    input.read_exact(&mut expected)?;
    if payload_len != bytes.len() as u64 - HEADER_LEN {
        return Err(DatabaseError::Format("stable-ID payload length"));
    }
    let payload = &bytes[HEADER_LEN as usize..];
    if md5::compute(payload).0 != expected {
        return Err(DatabaseError::Format("stable-ID checksum"));
    }
    let mut input = Cursor::new(payload);
    let count = read_count(&mut input)?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        entries.push(StableIdEntry {
            namespace: read_string(&mut input)?,
            name: read_string(&mut input)?,
            id: read_u64(&mut input)?,
        });
    }
    if input.position() != payload.len() as u64 {
        return Err(DatabaseError::Format("trailing stable-ID payload"));
    }
    validate_stable_ids(&entries)?;
    Ok(entries)
}

/// Dependency metadata for one future serialized compiler/runtime section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectionDependency {
    /// Stable section identity.
    pub section_id: u64,
    /// Sections that must be valid before this section can be reused.
    pub section_dependencies: Vec<u64>,
    /// Indices into [`PersistentCompilerDatabase::inputs`].
    pub input_dependencies: Vec<u64>,
    /// Digest of this section's linked representation.
    pub content_digest: Digest,
    /// Optional reusable linked-stage payload owned by this section.
    pub payload: Vec<u8>,
}

/// Persistent incremental compiler metadata, independent of runtime payloads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentCompilerDatabase {
    /// Digest of compiler semantics and schema-affecting implementation state.
    pub semantic_digest: Digest,
    /// Digest of macros, flags, target settings, and other build configuration.
    pub build_configuration_digest: Digest,
    /// Canonically ordered source, environment, map, and resource inputs.
    pub inputs: Vec<InputDependency>,
    /// Stable linker identity assignments.
    pub stable_ids: Vec<StableIdEntry>,
    /// Section-to-section and section-to-input dependency graph.
    pub sections: Vec<SectionDependency>,
}

impl PersistentCompilerDatabase {
    /// Reassembles an inline or paged section payload in dependency order.
    #[must_use]
    pub fn section_payload(&self, section_id: u64) -> Option<Vec<u8>> {
        let section = self
            .sections
            .iter()
            .find(|item| item.section_id == section_id)?;
        if !section.payload.is_empty() {
            return Some(section.payload.clone());
        }
        let by_id = self
            .sections
            .iter()
            .map(|item| (item.section_id, item))
            .collect::<std::collections::HashMap<_, _>>();
        let pages = section
            .section_dependencies
            .iter()
            .map(|id| {
                let page = *by_id.get(id)?;
                (!page.payload.is_empty()
                    && page.content_digest[..16] == md5::compute(&page.payload).0)
                    .then_some(page)
            })
            .collect::<Option<Vec<_>>>()?;
        if pages.is_empty() {
            return None;
        }
        let total = pages
            .iter()
            .try_fold(0_usize, |total, page| total.checked_add(page.payload.len()))?;
        let mut payload = Vec::with_capacity(total);
        for page in pages {
            payload.extend_from_slice(&page.payload);
        }
        Some(payload)
    }

    /// Replaces a section with independently bounded ordered payload pages.
    pub fn replace_paged_section(
        &mut self,
        section_id: u64,
        page_id_base: u64,
        content_digest: Digest,
        payload: &[u8],
        page_dependencies: Vec<u64>,
    ) -> Result<usize, DatabaseError> {
        let page_count = payload.len().div_ceil(MAX_SECTION_PAYLOAD_BYTES);
        let page_end = page_id_base
            .checked_add(1_000_000)
            .ok_or(DatabaseError::Limit("section page IDs"))?;
        self.sections.retain(|section| {
            section.section_id != section_id
                && !(section.section_id >= page_id_base && section.section_id < page_end)
        });
        let mut page_ids = Vec::with_capacity(page_count);
        for (index, page) in payload.chunks(MAX_SECTION_PAYLOAD_BYTES).enumerate() {
            let page_id = page_id_base + index as u64;
            page_ids.push(page_id);
            self.sections.push(SectionDependency {
                section_id: page_id,
                section_dependencies: page_dependencies.clone(),
                input_dependencies: vec![],
                content_digest: digest_page(page),
                payload: page.to_vec(),
            });
        }
        self.sections.push(SectionDependency {
            section_id,
            section_dependencies: page_ids,
            input_dependencies: vec![],
            content_digest,
            payload: vec![],
        });
        self.sections.sort_by_key(|section| section.section_id);
        validate(self)?;
        Ok(page_count)
    }
    /// Returns whether compiler semantics and build configuration are reusable.
    #[must_use]
    pub fn matches_build(&self, semantic: &Digest, configuration: &Digest) -> bool {
        self.semantic_digest == *semantic && self.build_configuration_digest == *configuration
    }

    /// Returns input indices whose identity, length, or digest changed.
    #[must_use]
    pub fn changed_inputs(&self, current: &[InputDependency]) -> Vec<u64> {
        let maximum = self.inputs.len().max(current.len());
        (0..maximum)
            .filter(|&index| self.inputs.get(index) != current.get(index))
            .map(|index| index as u64)
            .collect()
    }

    /// Returns section IDs invalidated by changed inputs or dependent sections.
    #[must_use]
    pub fn invalidated_sections(&self, changed_inputs: &[u64]) -> Vec<u64> {
        let changed: HashSet<_> = changed_inputs.iter().copied().collect();
        let mut invalid = HashSet::new();
        loop {
            let before = invalid.len();
            for section in &self.sections {
                if section
                    .input_dependencies
                    .iter()
                    .any(|index| changed.contains(index))
                    || section
                        .section_dependencies
                        .iter()
                        .any(|dependency| invalid.contains(dependency))
                {
                    invalid.insert(section.section_id);
                }
            }
            if invalid.len() == before {
                break;
            }
        }
        let mut result: Vec<_> = invalid.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Encodes and validates the bounded vNext database representation.
    pub fn encode(&self) -> Result<Vec<u8>, DatabaseError> {
        validate(self)?;
        let mut payload = Vec::new();
        payload.extend_from_slice(&self.semantic_digest);
        payload.extend_from_slice(&self.build_configuration_digest);
        write_u64(&mut payload, self.inputs.len() as u64);
        for input in &self.inputs {
            write_string(&mut payload, &input.identity)?;
            payload.extend_from_slice(&input.content_digest);
            write_u64(&mut payload, input.byte_length);
        }
        write_u64(&mut payload, self.stable_ids.len() as u64);
        for entry in &self.stable_ids {
            write_string(&mut payload, &entry.namespace)?;
            write_string(&mut payload, &entry.name)?;
            write_u64(&mut payload, entry.id);
        }
        write_u64(&mut payload, self.sections.len() as u64);
        for section in &self.sections {
            write_u64(&mut payload, section.section_id);
            write_u64_vec(&mut payload, &section.section_dependencies)?;
            write_u64_vec(&mut payload, &section.input_dependencies)?;
            payload.extend_from_slice(&section.content_digest);
            if section.payload.len() > MAX_SECTION_PAYLOAD_BYTES {
                return Err(DatabaseError::Limit("section payload"));
            }
            write_u64(&mut payload, section.payload.len() as u64);
            payload.extend_from_slice(&section.payload);
        }
        if payload.len() as u64 > MAX_DATABASE_BYTES - HEADER_LEN {
            return Err(DatabaseError::Limit("database payload"));
        }
        let checksum = md5::compute(&payload);
        let mut output = Vec::with_capacity(HEADER_LEN as usize + payload.len());
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(&VERSION.to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        write_u64(&mut output, payload.len() as u64);
        output.extend_from_slice(&checksum.0);
        output.extend_from_slice(&payload);
        Ok(output)
    }

    /// Decodes a bounded database and rejects malformed dependency metadata.
    pub fn decode(bytes: &[u8]) -> Result<Self, DatabaseError> {
        if bytes.len() as u64 > MAX_DATABASE_BYTES {
            return Err(DatabaseError::Limit("database file"));
        }
        if (bytes.len() as u64) < HEADER_LEN {
            return Err(DatabaseError::Format("truncated header"));
        }
        let mut input = Cursor::new(bytes);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(DatabaseError::Format("magic"));
        }
        if read_u32(&mut input)? != VERSION || read_u32(&mut input)? != 0 {
            return Err(DatabaseError::Format("version or flags"));
        }
        let payload_len = read_u64(&mut input)?;
        let mut expected = [0; 16];
        input.read_exact(&mut expected)?;
        if payload_len != bytes.len() as u64 - HEADER_LEN {
            return Err(DatabaseError::Format("payload length"));
        }
        let payload = &bytes[HEADER_LEN as usize..];
        if md5::compute(payload).0 != expected {
            return Err(DatabaseError::Format("checksum"));
        }
        let mut input = Cursor::new(payload);
        let semantic_digest = read_digest(&mut input)?;
        let build_configuration_digest = read_digest(&mut input)?;
        let input_count = read_count(&mut input)?;
        let mut inputs = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            inputs.push(InputDependency {
                identity: read_string(&mut input)?,
                content_digest: read_digest(&mut input)?,
                byte_length: read_u64(&mut input)?,
            });
        }
        let stable_id_count = read_count(&mut input)?;
        let mut stable_ids = Vec::with_capacity(stable_id_count);
        for _ in 0..stable_id_count {
            stable_ids.push(StableIdEntry {
                namespace: read_string(&mut input)?,
                name: read_string(&mut input)?,
                id: read_u64(&mut input)?,
            });
        }
        let section_count = read_count(&mut input)?;
        let mut sections = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            sections.push(SectionDependency {
                section_id: read_u64(&mut input)?,
                section_dependencies: read_u64_vec(&mut input)?,
                input_dependencies: read_u64_vec(&mut input)?,
                content_digest: read_digest(&mut input)?,
                payload: read_bytes(&mut input)?,
            });
        }
        if input.position() != payload.len() as u64 {
            return Err(DatabaseError::Format("trailing payload"));
        }
        let database = Self {
            semantic_digest,
            build_configuration_digest,
            inputs,
            stable_ids,
            sections,
        };
        validate(&database)?;
        Ok(database)
    }

    /// Reads and decodes one bounded database file.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, DatabaseError> {
        let path = path.as_ref();
        let length = fs::metadata(path)?.len();
        if length > MAX_DATABASE_BYTES {
            return Err(DatabaseError::Limit("database file"));
        }
        let mut bytes = Vec::with_capacity(length as usize);
        File::open(path)?
            .take(MAX_DATABASE_BYTES + 1)
            .read_to_end(&mut bytes)?;
        Self::decode(&bytes)
    }

    /// Writes a complete sibling temporary file and atomically installs it.
    pub fn write_atomic(&self, path: impl AsRef<Path>) -> Result<(), DatabaseError> {
        let path = path.as_ref();
        let bytes = self.encode()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let (temporary, mut file) = create_temporary_sibling(path)?;
        let result = (|| {
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }
}

fn digest_page(bytes: &[u8]) -> Digest {
    let first = md5::compute(bytes).0;
    let second = md5::compute(first).0;
    let mut digest = [0; 32];
    digest[..16].copy_from_slice(&first);
    digest[16..].copy_from_slice(&second);
    digest
}

/// Failure while validating or persisting a compiler database.
#[derive(Debug)]
pub enum DatabaseError {
    /// Filesystem or stream failure.
    Io(io::Error),
    /// Malformed or incompatible serialized data.
    Format(&'static str),
    /// A bounded collection or payload exceeded its limit.
    Limit(&'static str),
}

impl fmt::Display for DatabaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "compiler database I/O failed: {error}"),
            Self::Format(what) => write!(formatter, "invalid compiler database {what}"),
            Self::Limit(what) => write!(formatter, "compiler database {what} exceeds its limit"),
        }
    }
}

impl std::error::Error for DatabaseError {}

impl From<io::Error> for DatabaseError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn validate(database: &PersistentCompilerDatabase) -> Result<(), DatabaseError> {
    for length in [
        database.inputs.len(),
        database.stable_ids.len(),
        database.sections.len(),
    ] {
        if length as u64 > MAX_RECORDS {
            return Err(DatabaseError::Limit("record count"));
        }
    }
    let mut input_names = HashSet::new();
    if database
        .inputs
        .iter()
        .any(|item| !input_names.insert(&item.identity))
    {
        return Err(DatabaseError::Format("duplicate input identity"));
    }
    validate_stable_ids(&database.stable_ids)?;
    let section_ids: HashSet<_> = database
        .sections
        .iter()
        .map(|item| item.section_id)
        .collect();
    if section_ids.len() != database.sections.len() {
        return Err(DatabaseError::Format("duplicate section ID"));
    }
    for section in &database.sections {
        if section
            .input_dependencies
            .iter()
            .any(|&index| index >= database.inputs.len() as u64)
            || section
                .section_dependencies
                .iter()
                .any(|id| !section_ids.contains(id))
        {
            return Err(DatabaseError::Format("unknown dependency"));
        }
    }
    Ok(())
}

fn validate_stable_ids(entries: &[StableIdEntry]) -> Result<(), DatabaseError> {
    if entries.len() as u64 > MAX_RECORDS {
        return Err(DatabaseError::Limit("stable-ID record count"));
    }
    let mut stable_names = HashSet::new();
    let mut stable_numbers = HashSet::new();
    for entry in entries {
        if !stable_names.insert((&entry.namespace, &entry.name))
            || !stable_numbers.insert((&entry.namespace, entry.id))
        {
            return Err(DatabaseError::Format("duplicate stable ID"));
        }
    }
    Ok(())
}

fn normalize_identity(path: &Path) -> String {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.last().is_some_and(|part| part != "..") {
                    parts.pop();
                } else {
                    parts.push("..".to_owned());
                }
            }
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::RootDir => parts.push(String::new()),
            Component::Prefix(prefix) => {
                parts.push(prefix.as_os_str().to_string_lossy().replace('\\', "/"))
            }
        }
    }
    parts.join("/")
}

fn create_temporary_sibling(path: &Path) -> Result<(PathBuf, File), DatabaseError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or(DatabaseError::Format("output filename"))?;
    for _ in 0..128 {
        let sequence = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{}.{}.tmp", name.to_string_lossy(), sequence));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    Err(DatabaseError::Limit("temporary-file attempts"))
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<(), DatabaseError> {
    if value.len() as u64 > MAX_STRING_BYTES {
        return Err(DatabaseError::Limit("string"));
    }
    write_u64(output, value.len() as u64);
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_u64_vec(output: &mut Vec<u8>, values: &[u64]) -> Result<(), DatabaseError> {
    if values.len() as u64 > MAX_RECORDS {
        return Err(DatabaseError::Limit("dependency count"));
    }
    write_u64(output, values.len() as u64);
    for &value in values {
        write_u64(output, value);
    }
    Ok(())
}

fn write_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(input: &mut Cursor<&[u8]>) -> Result<u32, DatabaseError> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(input: &mut Cursor<&[u8]>) -> Result<u64, DatabaseError> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_count(input: &mut Cursor<&[u8]>) -> Result<usize, DatabaseError> {
    let count = read_u64(input)?;
    if count > MAX_RECORDS {
        return Err(DatabaseError::Limit("record count"));
    }
    usize::try_from(count).map_err(|_| DatabaseError::Limit("native record count"))
}

fn read_string(input: &mut Cursor<&[u8]>) -> Result<String, DatabaseError> {
    let length = read_u64(input)?;
    if length > MAX_STRING_BYTES {
        return Err(DatabaseError::Limit("string"));
    }
    let mut bytes = vec![0; usize::try_from(length).map_err(|_| DatabaseError::Limit("string"))?];
    input.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| DatabaseError::Format("UTF-8 string"))
}

fn read_u64_vec(input: &mut Cursor<&[u8]>) -> Result<Vec<u64>, DatabaseError> {
    let count = read_count(input)?;
    (0..count).map(|_| read_u64(input)).collect()
}

fn read_digest(input: &mut Cursor<&[u8]>) -> Result<Digest, DatabaseError> {
    let mut digest = [0; 32];
    input.read_exact(&mut digest)?;
    Ok(digest)
}

fn read_bytes(input: &mut Cursor<&[u8]>) -> Result<Vec<u8>, DatabaseError> {
    let length = read_u64(input)?;
    if length > MAX_SECTION_PAYLOAD_BYTES as u64 {
        return Err(DatabaseError::Limit("section payload"));
    }
    let mut bytes =
        vec![0; usize::try_from(length).map_err(|_| DatabaseError::Limit("section payload"))?];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PersistentCompilerDatabase {
        PersistentCompilerDatabase {
            semantic_digest: [1; 32],
            build_configuration_digest: [2; 32],
            inputs: vec![
                InputDependency::new("./code/../code/world.dm", [3; 32], 42),
                InputDependency::new("icons\\hud.dmi", [4; 32], 900),
            ],
            stable_ids: vec![StableIdEntry {
                namespace: "type".to_owned(),
                name: "/datum/example".to_owned(),
                id: u64::from(u32::MAX) + 7,
            }],
            sections: vec![
                SectionDependency {
                    section_id: 10,
                    section_dependencies: vec![],
                    input_dependencies: vec![0],
                    content_digest: [5; 32],
                    payload: vec![1, 2, 3],
                },
                SectionDependency {
                    section_id: 20,
                    section_dependencies: vec![10],
                    input_dependencies: vec![1],
                    content_digest: [6; 32],
                    payload: vec![],
                },
            ],
        }
    }

    #[test]
    fn database_round_trips_with_native_64_bit_fields() {
        let database = fixture();
        let encoded = database.encode().unwrap();
        let decoded = PersistentCompilerDatabase::decode(&encoded).unwrap();
        assert_eq!(decoded, database);
        assert_eq!(decoded.inputs[0].identity, "code/world.dm");
        assert!(decoded.stable_ids[0].id > u64::from(u32::MAX));

        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(PersistentCompilerDatabase::decode(&corrupt).is_err());
    }

    #[test]
    fn stable_id_artifact_table_round_trips_and_rejects_corruption() {
        let entries = fixture().stable_ids;
        let encoded = encode_stable_id_table(&entries).unwrap();
        assert_eq!(decode_stable_id_table(&encoded).unwrap(), entries);
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(decode_stable_id_table(&corrupt).is_err());
    }

    #[test]
    fn paged_section_rejects_missing_or_corrupt_pages() {
        let mut database = fixture();
        database
            .replace_paged_section(30, 50_000, [9; 32], &[1, 2, 3, 4], vec![10])
            .unwrap();
        assert_eq!(database.section_payload(30).unwrap(), [1, 2, 3, 4]);
        let page = database
            .sections
            .iter_mut()
            .find(|section| section.section_id == 50_000)
            .unwrap();
        page.payload[0] ^= 1;
        assert!(database.section_payload(30).is_none());
        database
            .sections
            .retain(|section| section.section_id != 50_000);
        assert!(database.section_payload(30).is_none());
    }

    #[test]
    fn build_and_dependency_invalidation_is_transitive() {
        let database = fixture();
        assert!(database.matches_build(&[1; 32], &[2; 32]));
        assert!(!database.matches_build(&[9; 32], &[2; 32]));

        let mut current = database.inputs.clone();
        current[0].content_digest = [8; 32];
        let changed = database.changed_inputs(&current);
        assert_eq!(changed, vec![0]);
        assert_eq!(database.invalidated_sections(&changed), vec![10, 20]);
    }

    #[test]
    fn atomic_file_roundtrip_rejects_unknown_dependencies() {
        let directory = std::env::temp_dir().join(format!(
            "dream64-cdb-test-{}",
            NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed)
        ));
        let path = directory.join("project.d64cdb");
        let database = fixture();
        database.write_atomic(&path).unwrap();
        assert_eq!(PersistentCompilerDatabase::read(&path).unwrap(), database);
        let mut replacement = database.clone();
        replacement.build_configuration_digest = [7; 32];
        replacement.write_atomic(&path).unwrap();
        assert_eq!(
            PersistentCompilerDatabase::read(&path).unwrap(),
            replacement
        );

        let mut invalid = database;
        invalid.sections[0].input_dependencies = vec![99];
        assert!(invalid.encode().is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}
