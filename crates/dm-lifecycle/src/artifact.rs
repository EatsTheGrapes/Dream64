//! Durable envelope for compiled Dream64 artifacts.
//!
//! The payload is deliberately opaque here. Compiler/module codecs can assign
//! stable section identifiers without coupling persistence safety to a
//! particular in-memory representation.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

include!(concat!(env!("OUT_DIR"), "/engine_semantics_fingerprint.rs"));

const MAGIC: &[u8; 16] = b"DREAM64-COMPILED";
const COMMIT_MAGIC: &[u8; 16] = b"DREAM64-COMMIT!!";
const FIXED_HEADER_LENGTH: usize = 72;
const SECTION_TABLE_ENTRY_LENGTH: usize = 24;
const HEADER_CHECKSUM_LENGTH: usize = 4;
const FOOTER_LENGTH: usize = 32;
const MAX_TARGET_LENGTH: usize = 255;
const DEFAULT_MMAP_THRESHOLD: u64 = 1024 * 1024;

/// Current on-disk artifact schema.
pub const ARTIFACT_SCHEMA: u16 = 1;

/// Maximum accepted encoded artifact length (16 GiB).
pub const MAX_ARTIFACT_LENGTH: u64 = 16 * 1024 * 1024 * 1024;

/// Maximum accepted payload section length (8 GiB).
pub const MAX_SECTION_LENGTH: u64 = 8 * 1024 * 1024 * 1024;

/// Maximum number of independently checksummed payload sections.
pub const MAX_SECTIONS: usize = 1_024;

static TEMPORARY_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Returns the build-derived identity of code which determines compiled DM
/// semantics.
///
/// Cargo regenerates this value from the compiler, lowering, semantics, VM,
/// and runtime source trees. A change to any of those inputs therefore makes
/// an older artifact ineligible for loading.
#[must_use]
pub const fn engine_semantics_fingerprint() -> [u8; 16] {
    GENERATED_ENGINE_SEMANTICS_FINGERPRINT
}

/// One opaque, independently checksummed payload in a compiled artifact.
#[derive(Clone)]
pub struct ArtifactSection {
    id: u32,
    payload: ArtifactPayload,
}

#[derive(Clone)]
enum ArtifactPayload {
    Owned(Vec<u8>),
    Shared {
        backing: Arc<Vec<u8>>,
        offset: usize,
        length: usize,
    },
    Mapped {
        backing: Arc<dm_mmap::ReadOnlyMapping>,
        offset: usize,
        length: usize,
    },
}

impl ArtifactSection {
    /// Creates a section with a codec-owned stable numeric identifier.
    #[must_use]
    pub fn new(id: u32, payload: Vec<u8>) -> Self {
        Self {
            id,
            payload: ArtifactPayload::Owned(payload),
        }
    }

    /// Returns the codec-owned section identifier.
    #[must_use]
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// Returns the opaque encoded section bytes.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        match &self.payload {
            ArtifactPayload::Owned(payload) => payload,
            ArtifactPayload::Shared {
                backing,
                offset,
                length,
            } => &backing[*offset..*offset + *length],
            ArtifactPayload::Mapped {
                backing,
                offset,
                length,
            } => &backing[*offset..*offset + *length],
        }
    }

    /// Consumes the section and returns its opaque encoded bytes.
    #[must_use]
    pub fn into_payload(self) -> Vec<u8> {
        match self.payload {
            ArtifactPayload::Owned(payload) => payload,
            ArtifactPayload::Shared {
                backing,
                offset,
                length,
            } => backing[offset..offset + length].to_vec(),
            ArtifactPayload::Mapped {
                backing,
                offset,
                length,
            } => backing[offset..offset + length].to_vec(),
        }
    }

    fn from_shared(id: u32, backing: Arc<Vec<u8>>, offset: usize, length: usize) -> Self {
        Self {
            id,
            payload: ArtifactPayload::Shared {
                backing,
                offset,
                length,
            },
        }
    }

    fn from_mapped(
        id: u32,
        backing: Arc<dm_mmap::ReadOnlyMapping>,
        offset: usize,
        length: usize,
    ) -> Self {
        Self {
            id,
            payload: ArtifactPayload::Mapped {
                backing,
                offset,
                length,
            },
        }
    }
}

impl fmt::Debug for ArtifactSection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactSection")
            .field("id", &self.id)
            .field("payload", &self.payload())
            .finish()
    }
}

impl PartialEq for ArtifactSection {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.payload() == other.payload()
    }
}

impl Eq for ArtifactSection {}

/// Physical storage retained by one decoded artifact.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArtifactStorageStats {
    /// Sum of logical section payload lengths.
    pub payload_bytes: usize,
    /// Bytes retained by distinct backing allocations.
    pub backing_bytes: usize,
    /// Number of distinct backing allocations.
    pub backing_allocations: usize,
    /// Distinct retained read-only file mappings.
    pub mapped_backings: usize,
    /// Distinct retained heap buffers shared by sections.
    pub buffered_backings: usize,
}

/// File-read policy for compiled artifacts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactReadMode {
    /// Map files at least 1 MiB and buffer smaller files.
    Auto,
    /// Always use the bounded buffered path.
    Buffered,
    /// Attempt a read-only mapping regardless of size, with buffered fallback.
    PreferMapped,
}

/// Bounded-memory characteristics of one atomic artifact installation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArtifactWriteStats {
    /// Complete committed file length.
    pub encoded_bytes: u64,
    /// Section bytes streamed from their existing codec-owned buffers.
    pub payload_bytes: u64,
    /// Largest additional envelope buffer retained while writing.
    pub peak_staging_bytes: usize,
    /// Header, section, and footer writes issued to the temporary file.
    pub write_calls: usize,
}

/// A validated collection of compiled payload sections for exactly one
/// project input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledArtifact {
    project_fingerprint: [u8; 16],
    sections: Vec<ArtifactSection>,
}

impl CompiledArtifact {
    /// Builds an artifact from opaque payload sections.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate identifiers, oversized sections, too
    /// many sections, or an aggregate encoding larger than the format bound.
    pub fn new(
        project_fingerprint: [u8; 16],
        sections: Vec<ArtifactSection>,
    ) -> Result<Self, ArtifactError> {
        validate_sections(&sections)?;
        let artifact = Self {
            project_fingerprint,
            sections,
        };
        artifact.encoded_lengths(current_target().len())?;
        Ok(artifact)
    }

    /// Returns the exact compiler-visible project content identity.
    #[must_use]
    pub const fn project_fingerprint(&self) -> [u8; 16] {
        self.project_fingerprint
    }

    /// Returns payload sections in their encoded order.
    #[must_use]
    pub fn sections(&self) -> &[ArtifactSection] {
        &self.sections
    }

    /// Returns the first section with the requested codec identifier.
    #[must_use]
    pub fn section(&self, id: u32) -> Option<&ArtifactSection> {
        self.sections.iter().find(|section| section.id == id)
    }

    /// Reports logical payload and physical backing storage retained in memory.
    #[must_use]
    pub fn storage_stats(&self) -> ArtifactStorageStats {
        let mut stats = ArtifactStorageStats::default();
        let mut shared = BTreeSet::new();
        for section in &self.sections {
            stats.payload_bytes = stats.payload_bytes.saturating_add(section.payload().len());
            match &section.payload {
                ArtifactPayload::Owned(payload) => {
                    stats.backing_bytes = stats.backing_bytes.saturating_add(payload.capacity());
                    stats.backing_allocations += 1;
                }
                ArtifactPayload::Shared { backing, .. } => {
                    let identity = Arc::as_ptr(backing) as usize;
                    if shared.insert(identity) {
                        stats.backing_bytes =
                            stats.backing_bytes.saturating_add(backing.capacity());
                        stats.backing_allocations += 1;
                        stats.buffered_backings += 1;
                    }
                }
                ArtifactPayload::Mapped { backing, .. } => {
                    let identity = Arc::as_ptr(backing) as usize;
                    if shared.insert(identity) {
                        stats.backing_bytes = stats.backing_bytes.saturating_add(backing.len());
                        stats.backing_allocations += 1;
                        stats.mapped_backings += 1;
                    }
                }
            }
        }
        stats
    }

    /// Encodes the complete, committed artifact deterministically.
    ///
    /// # Errors
    ///
    /// Returns an error if current contents exceed a format bound.
    pub fn encode(&self) -> Result<Vec<u8>, ArtifactError> {
        self.encode_with_metadata(EncodingMetadata::current())
    }

    /// Decodes and validates an artifact for an expected project.
    ///
    /// Validation covers the schema, target/endian marker, engine semantics,
    /// project identity, canonical section layout, every CRC32, exact total
    /// length, and the committed footer.
    ///
    /// # Errors
    ///
    /// Returns a specific mismatch, bounds, layout, checksum, truncation, or
    /// uncommitted-artifact error. No payload is exposed before all checks pass.
    pub fn decode(
        bytes: &[u8],
        expected_project_fingerprint: [u8; 16],
    ) -> Result<Self, ArtifactError> {
        decode_artifact(bytes, expected_project_fingerprint)
    }

    /// Reads and fully validates an artifact for an expected project.
    ///
    /// The file length is bounded before allocating its buffer and is checked
    /// again by the decoder, protecting against both malformed metadata and a
    /// concurrent file-size change.
    ///
    /// # Errors
    ///
    /// Returns file I/O errors or any validation error returned by
    /// [`Self::decode`].
    pub fn read_from(
        path: impl AsRef<Path>,
        expected_project_fingerprint: [u8; 16],
    ) -> Result<Self, ArtifactError> {
        Self::read_from_with_mode(path, expected_project_fingerprint, ArtifactReadMode::Auto)
    }

    /// Reads with an explicit mapping/buffering policy for diagnostics and
    /// reproducible benchmarks. Mapping errors always fall back to buffering.
    pub fn read_from_with_mode(
        path: impl AsRef<Path>,
        expected_project_fingerprint: [u8; 16],
        mode: ArtifactReadMode,
    ) -> Result<Self, ArtifactError> {
        read_from_with_mapper(
            path.as_ref(),
            expected_project_fingerprint,
            mode,
            dm_mmap::ReadOnlyMapping::map,
        )
    }

    #[cfg(test)]
    fn read_from_with_mapper_for_test(
        path: &Path,
        expected_project_fingerprint: [u8; 16],
        mode: ArtifactReadMode,
        mapper: impl FnOnce(&File) -> io::Result<dm_mmap::ReadOnlyMapping>,
    ) -> Result<Self, ArtifactError> {
        read_from_with_mapper(path, expected_project_fingerprint, mode, mapper)
    }

    /// Atomically installs a fully committed artifact at `path`.
    ///
    /// Bytes are written to a uniquely created sibling file, flushed, synced
    /// to storage, and only then renamed over the destination. A failed write
    /// leaves the previous destination untouched and removes the temporary
    /// file on a best-effort basis.
    ///
    /// # Errors
    ///
    /// Returns encoding or file I/O errors.
    pub fn write_atomic(&self, path: impl AsRef<Path>) -> Result<(), ArtifactError> {
        self.write_atomic_with_stats(path).map(|_| ())
    }

    /// Atomically installs an artifact while reporting its bounded staging
    /// footprint.
    ///
    /// Section payloads are streamed directly from their existing buffers. The
    /// writer never assembles a second artifact-sized contiguous allocation.
    ///
    /// # Errors
    ///
    /// Returns encoding or file I/O errors.
    pub fn write_atomic_with_stats(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ArtifactWriteStats, ArtifactError> {
        let path = path.as_ref();
        let (header, total_length) =
            self.encode_header_with_metadata(EncodingMetadata::current())?;
        let total_u64 = u64::try_from(total_length).map_err(|_| ArtifactError::LengthOverflow)?;
        let payload_bytes = self.sections.iter().try_fold(0_u64, |length, section| {
            length
                .checked_add(
                    u64::try_from(section.payload().len())
                        .map_err(|_| ArtifactError::LengthOverflow)?,
                )
                .ok_or(ArtifactError::LengthOverflow)
        })?;
        let stats = ArtifactWriteStats {
            encoded_bytes: total_u64,
            payload_bytes,
            peak_staging_bytes: header.capacity().max(FOOTER_LENGTH),
            write_calls: self.sections.len().saturating_add(2),
        };
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|source| ArtifactError::Io {
            operation: "create parent directory for",
            path: path.to_path_buf(),
            source,
        })?;
        let (temporary_path, mut temporary_file) = create_temporary_sibling(path)?;
        let write_result = (|| {
            let mut body_checksum = crc32fast::Hasher::new();
            body_checksum.update(&header);
            temporary_file
                .write_all(&header)
                .map_err(|source| ArtifactError::Io {
                    operation: "write temporary artifact header",
                    path: temporary_path.clone(),
                    source,
                })?;
            for section in &self.sections {
                let payload = section.payload();
                body_checksum.update(payload);
                temporary_file
                    .write_all(payload)
                    .map_err(|source| ArtifactError::Io {
                        operation: "write temporary artifact section",
                        path: temporary_path.clone(),
                        source,
                    })?;
            }
            let footer = encode_footer(total_u64, body_checksum.finalize());
            temporary_file
                .write_all(&footer)
                .map_err(|source| ArtifactError::Io {
                    operation: "write temporary artifact footer",
                    path: temporary_path.clone(),
                    source,
                })?;
            temporary_file
                .flush()
                .and_then(|()| temporary_file.sync_all())
                .map_err(|source| ArtifactError::Io {
                    operation: "flush and sync temporary artifact",
                    path: temporary_path.clone(),
                    source,
                })?;
            drop(temporary_file);
            fs::rename(&temporary_path, path).map_err(|source| ArtifactError::Io {
                operation: "atomically replace",
                path: path.to_path_buf(),
                source,
            })?;
            // Directory syncing is supported on Unix and some other targets,
            // but not uniformly. The artifact file itself has already been
            // synced; make the directory durability barrier when available.
            if let Ok(directory) = File::open(parent) {
                let _ = directory.sync_all();
            }
            Ok(stats)
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temporary_path);
        }
        write_result
    }

    fn encoded_lengths(&self, target_length: usize) -> Result<(usize, usize), ArtifactError> {
        if target_length > MAX_TARGET_LENGTH {
            return Err(ArtifactError::TargetTooLong {
                actual: target_length,
                maximum: MAX_TARGET_LENGTH,
            });
        }
        validate_sections(&self.sections)?;
        let table_length = self
            .sections
            .len()
            .checked_mul(SECTION_TABLE_ENTRY_LENGTH)
            .ok_or(ArtifactError::LengthOverflow)?;
        let header_length = FIXED_HEADER_LENGTH
            .checked_add(target_length)
            .and_then(|length| length.checked_add(table_length))
            .and_then(|length| length.checked_add(HEADER_CHECKSUM_LENGTH))
            .ok_or(ArtifactError::LengthOverflow)?;
        let payload_length = self.sections.iter().try_fold(0_usize, |length, section| {
            length
                .checked_add(section.payload().len())
                .ok_or(ArtifactError::LengthOverflow)
        })?;
        let total_length = header_length
            .checked_add(payload_length)
            .and_then(|length| length.checked_add(FOOTER_LENGTH))
            .ok_or(ArtifactError::LengthOverflow)?;
        let total_u64 = u64::try_from(total_length).map_err(|_| ArtifactError::LengthOverflow)?;
        if total_u64 > MAX_ARTIFACT_LENGTH {
            return Err(ArtifactError::ArtifactTooLarge {
                actual: total_u64,
                maximum: MAX_ARTIFACT_LENGTH,
            });
        }
        Ok((header_length, total_length))
    }

    fn encode_with_metadata(
        &self,
        metadata: EncodingMetadata<'_>,
    ) -> Result<Vec<u8>, ArtifactError> {
        let (header, total_length) = self.encode_header_with_metadata(metadata)?;
        let total_u64 = u64::try_from(total_length).map_err(|_| ArtifactError::LengthOverflow)?;
        let mut encoded = Vec::with_capacity(total_length);
        encoded.extend_from_slice(&header);
        for section in &self.sections {
            encoded.extend_from_slice(section.payload());
        }
        debug_assert_eq!(encoded.len() + FOOTER_LENGTH, total_length);

        let body_checksum = crc32fast::hash(&encoded);
        encoded.extend_from_slice(&encode_footer(total_u64, body_checksum));
        debug_assert_eq!(encoded.len(), total_length);
        Ok(encoded)
    }

    fn encode_header_with_metadata(
        &self,
        metadata: EncodingMetadata<'_>,
    ) -> Result<(Vec<u8>, usize), ArtifactError> {
        let (header_length, total_length) = self.encoded_lengths(metadata.target.len())?;
        let header_u32 = u32::try_from(header_length).map_err(|_| ArtifactError::LengthOverflow)?;
        let total_u64 = u64::try_from(total_length).map_err(|_| ArtifactError::LengthOverflow)?;
        let target_length =
            u16::try_from(metadata.target.len()).map_err(|_| ArtifactError::TargetTooLong {
                actual: metadata.target.len(),
                maximum: MAX_TARGET_LENGTH,
            })?;
        let section_count =
            u32::try_from(self.sections.len()).map_err(|_| ArtifactError::TooManySections {
                actual: self.sections.len(),
                maximum: MAX_SECTIONS,
            })?;

        let mut encoded = Vec::with_capacity(header_length);
        encoded.extend_from_slice(MAGIC);
        push_u16(&mut encoded, metadata.schema);
        encoded.push(metadata.endian);
        encoded.push(metadata.pointer_width);
        push_u16(&mut encoded, target_length);
        push_u16(&mut encoded, 0);
        push_u32(&mut encoded, section_count);
        push_u32(&mut encoded, header_u32);
        push_u64(&mut encoded, total_u64);
        encoded.extend_from_slice(&self.project_fingerprint);
        encoded.extend_from_slice(&metadata.engine_fingerprint);
        encoded.extend_from_slice(metadata.target);

        let mut payload_offset = header_length;
        for section in &self.sections {
            let section_length = section.payload().len();
            push_u32(&mut encoded, section.id);
            push_u64(
                &mut encoded,
                u64::try_from(payload_offset).map_err(|_| ArtifactError::LengthOverflow)?,
            );
            push_u64(
                &mut encoded,
                u64::try_from(section_length).map_err(|_| ArtifactError::LengthOverflow)?,
            );
            push_u32(&mut encoded, crc32fast::hash(section.payload()));
            payload_offset = payload_offset
                .checked_add(section_length)
                .ok_or(ArtifactError::LengthOverflow)?;
        }
        let header_checksum = crc32fast::hash(&encoded);
        push_u32(&mut encoded, header_checksum);
        debug_assert_eq!(encoded.len(), header_length);
        Ok((encoded, total_length))
    }
}

fn encode_footer(total_length: u64, body_checksum: u32) -> Vec<u8> {
    let mut footer = Vec::with_capacity(FOOTER_LENGTH);
    footer.extend_from_slice(COMMIT_MAGIC);
    push_u32(&mut footer, body_checksum);
    push_u64(&mut footer, total_length);
    let footer_checksum = crc32fast::hash(&footer);
    push_u32(&mut footer, footer_checksum);
    debug_assert_eq!(footer.len(), FOOTER_LENGTH);
    footer
}

/// Region whose stored CRC32 does not match its encoded bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChecksumRegion {
    /// Header metadata and section table.
    Header,
    /// Complete header and payload body.
    Body,
    /// Commit footer metadata.
    Footer,
    /// One payload section, identified by its stable codec ID.
    Section(u32),
}

/// Rejection reason for an encoded compiled artifact.
#[derive(Debug)]
#[non_exhaustive]
pub enum ArtifactError {
    /// File or directory operation failed.
    Io {
        /// Operation which failed.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Underlying operating-system error.
        source: io::Error,
    },
    /// The artifact exceeds the global encoded-size bound.
    ArtifactTooLarge {
        /// Observed encoded length.
        actual: u64,
        /// Maximum accepted encoded length.
        maximum: u64,
    },
    /// A payload section exceeds the per-section bound.
    SectionTooLarge {
        /// Stable section identifier.
        id: u32,
        /// Observed section length.
        actual: u64,
        /// Maximum accepted section length.
        maximum: u64,
    },
    /// More payload sections were supplied than the decoder permits.
    TooManySections {
        /// Observed section count.
        actual: usize,
        /// Maximum accepted section count.
        maximum: usize,
    },
    /// Two payload sections have the same stable identifier.
    DuplicateSection {
        /// Repeated stable section identifier.
        id: u32,
    },
    /// The encoded target identifier exceeds its format bound.
    TargetTooLong {
        /// Observed target identifier length.
        actual: usize,
        /// Maximum accepted target identifier length.
        maximum: usize,
    },
    /// Integer arithmetic could not represent an encoded length.
    LengthOverflow,
    /// The leading artifact magic is absent.
    InvalidMagic,
    /// The artifact uses a different storage schema.
    SchemaMismatch {
        /// Schema recorded by the artifact.
        found: u16,
        /// Schema supported by this engine.
        expected: u16,
    },
    /// The artifact belongs to a different project input.
    ProjectFingerprintMismatch,
    /// The artifact was compiled under different engine semantics.
    EngineFingerprintMismatch,
    /// The artifact targets a different Rust platform.
    TargetMismatch {
        /// Target recorded by the artifact.
        found: String,
        /// Target of the running engine.
        expected: &'static str,
    },
    /// The artifact was encoded for a different byte order.
    EndianMismatch,
    /// The artifact was encoded for a different pointer width.
    PointerWidthMismatch,
    /// The encoded file ended before its declared total length.
    Truncated {
        /// Length declared in the artifact header.
        expected: u64,
        /// Bytes actually supplied.
        actual: u64,
    },
    /// Bytes follow the declared committed artifact.
    ExtraData {
        /// Length declared in the artifact header.
        expected: u64,
        /// Bytes actually supplied.
        actual: u64,
    },
    /// Header, table, or payload offsets are not canonical and bounded.
    MalformedLayout {
        /// Stable description of the rejected invariant.
        reason: &'static str,
    },
    /// The terminal committed marker is missing.
    MissingCommitFooter,
    /// A CRC32 did not validate.
    ChecksumMismatch {
        /// Corrupted encoded region.
        region: ChecksumRegion,
    },
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "failed to {operation} {}: {source}",
                path.display()
            ),
            Self::ArtifactTooLarge { actual, maximum } => write!(
                formatter,
                "artifact length {actual} exceeds maximum {maximum}"
            ),
            Self::SectionTooLarge {
                id,
                actual,
                maximum,
            } => write!(
                formatter,
                "artifact section {id} length {actual} exceeds maximum {maximum}"
            ),
            Self::TooManySections { actual, maximum } => write!(
                formatter,
                "artifact has {actual} sections, exceeding maximum {maximum}"
            ),
            Self::DuplicateSection { id } => {
                write!(formatter, "artifact contains duplicate section {id}")
            }
            Self::TargetTooLong { actual, maximum } => write!(
                formatter,
                "artifact target length {actual} exceeds maximum {maximum}"
            ),
            Self::LengthOverflow => formatter.write_str("artifact length arithmetic overflowed"),
            Self::InvalidMagic => formatter.write_str("invalid compiled artifact magic"),
            Self::SchemaMismatch { found, expected } => write!(
                formatter,
                "compiled artifact schema {found} does not match engine schema {expected}"
            ),
            Self::ProjectFingerprintMismatch => {
                formatter.write_str("compiled artifact project fingerprint mismatch")
            }
            Self::EngineFingerprintMismatch => {
                formatter.write_str("compiled artifact engine semantics mismatch")
            }
            Self::TargetMismatch { found, expected } => write!(
                formatter,
                "compiled artifact target {found} does not match engine target {expected}"
            ),
            Self::EndianMismatch => formatter.write_str("compiled artifact endian mismatch"),
            Self::PointerWidthMismatch => {
                formatter.write_str("compiled artifact pointer-width mismatch")
            }
            Self::Truncated { expected, actual } => write!(
                formatter,
                "compiled artifact is truncated: expected {expected} bytes, found {actual}"
            ),
            Self::ExtraData { expected, actual } => write!(
                formatter,
                "compiled artifact has trailing data: expected {expected} bytes, found {actual}"
            ),
            Self::MalformedLayout { reason } => {
                write!(formatter, "malformed compiled artifact layout: {reason}")
            }
            Self::MissingCommitFooter => {
                formatter.write_str("compiled artifact has no committed footer")
            }
            Self::ChecksumMismatch { region } => {
                write!(formatter, "compiled artifact {region:?} checksum mismatch")
            }
        }
    }
}

impl std::error::Error for ArtifactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct EncodingMetadata<'a> {
    schema: u16,
    endian: u8,
    pointer_width: u8,
    target: &'a [u8],
    engine_fingerprint: [u8; 16],
}

impl EncodingMetadata<'static> {
    const fn current() -> Self {
        Self {
            schema: ARTIFACT_SCHEMA,
            endian: current_endian(),
            pointer_width: current_pointer_width(),
            target: current_target().as_bytes(),
            engine_fingerprint: GENERATED_ENGINE_SEMANTICS_FINGERPRINT,
        }
    }
}

#[derive(Clone, Copy)]
struct EncodedSection {
    id: u32,
    offset: usize,
    length: usize,
    checksum: u32,
}

struct ValidatedArtifact {
    project_fingerprint: [u8; 16],
    sections: Vec<EncodedSection>,
}

#[allow(clippy::too_many_lines)]
fn decode_artifact(
    bytes: &[u8],
    expected_project_fingerprint: [u8; 16],
) -> Result<CompiledArtifact, ArtifactError> {
    let validated = validate_artifact(bytes, expected_project_fingerprint)?;
    let sections = validated
        .sections
        .into_iter()
        .map(|section| {
            ArtifactSection::new(
                section.id,
                bytes[section.offset..section.offset + section.length].to_vec(),
            )
        })
        .collect();
    Ok(CompiledArtifact {
        project_fingerprint: validated.project_fingerprint,
        sections,
    })
}

fn decode_owned_artifact(
    bytes: Vec<u8>,
    expected_project_fingerprint: [u8; 16],
) -> Result<CompiledArtifact, ArtifactError> {
    let validated = validate_artifact(&bytes, expected_project_fingerprint)?;
    let backing = Arc::new(bytes);
    let sections = validated
        .sections
        .into_iter()
        .map(|section| {
            ArtifactSection::from_shared(
                section.id,
                Arc::clone(&backing),
                section.offset,
                section.length,
            )
        })
        .collect();
    Ok(CompiledArtifact {
        project_fingerprint: validated.project_fingerprint,
        sections,
    })
}

fn decode_mapped_artifact(
    backing: dm_mmap::ReadOnlyMapping,
    expected_project_fingerprint: [u8; 16],
) -> Result<CompiledArtifact, ArtifactError> {
    let validated = validate_artifact(&backing, expected_project_fingerprint)?;
    let backing = Arc::new(backing);
    let sections = validated
        .sections
        .into_iter()
        .map(|section| {
            ArtifactSection::from_mapped(
                section.id,
                Arc::clone(&backing),
                section.offset,
                section.length,
            )
        })
        .collect();
    Ok(CompiledArtifact {
        project_fingerprint: validated.project_fingerprint,
        sections,
    })
}

fn read_from_with_mapper(
    path: &Path,
    expected_project_fingerprint: [u8; 16],
    mode: ArtifactReadMode,
    mapper: impl FnOnce(&File) -> io::Result<dm_mmap::ReadOnlyMapping>,
) -> Result<CompiledArtifact, ArtifactError> {
    let file = File::open(path).map_err(|source| ArtifactError::Io {
        operation: "open",
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| ArtifactError::Io {
        operation: "inspect",
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > MAX_ARTIFACT_LENGTH {
        return Err(ArtifactError::ArtifactTooLarge {
            actual: metadata.len(),
            maximum: MAX_ARTIFACT_LENGTH,
        });
    }
    let should_map = match mode {
        ArtifactReadMode::Auto => metadata.len() >= DEFAULT_MMAP_THRESHOLD,
        ArtifactReadMode::Buffered => false,
        ArtifactReadMode::PreferMapped => true,
    };
    if should_map && let Ok(mapping) = mapper(&file) {
        return decode_mapped_artifact(mapping, expected_project_fingerprint);
    }
    drop(file);
    let bytes = fs::read(path).map_err(|source| ArtifactError::Io {
        operation: "read",
        path: path.to_path_buf(),
        source,
    })?;
    decode_owned_artifact(bytes, expected_project_fingerprint)
}

#[allow(clippy::too_many_lines)]
fn validate_artifact(
    bytes: &[u8],
    expected_project_fingerprint: [u8; 16],
) -> Result<ValidatedArtifact, ArtifactError> {
    let actual_length = u64::try_from(bytes.len()).map_err(|_| ArtifactError::LengthOverflow)?;
    if actual_length > MAX_ARTIFACT_LENGTH {
        return Err(ArtifactError::ArtifactTooLarge {
            actual: actual_length,
            maximum: MAX_ARTIFACT_LENGTH,
        });
    }
    let minimum_length = FIXED_HEADER_LENGTH
        .checked_add(HEADER_CHECKSUM_LENGTH)
        .and_then(|length| length.checked_add(FOOTER_LENGTH))
        .ok_or(ArtifactError::LengthOverflow)?;
    if bytes.len() < minimum_length {
        return Err(ArtifactError::Truncated {
            expected: u64::try_from(minimum_length).map_err(|_| ArtifactError::LengthOverflow)?,
            actual: actual_length,
        });
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(ArtifactError::InvalidMagic);
    }
    let schema = read_u16(bytes, 16)?;
    if schema != ARTIFACT_SCHEMA {
        return Err(ArtifactError::SchemaMismatch {
            found: schema,
            expected: ARTIFACT_SCHEMA,
        });
    }
    let encoded_endian = bytes[18];
    let encoded_pointer_width = bytes[19];
    let target_length = usize::from(read_u16(bytes, 20)?);
    if target_length > MAX_TARGET_LENGTH {
        return Err(ArtifactError::TargetTooLong {
            actual: target_length,
            maximum: MAX_TARGET_LENGTH,
        });
    }
    if read_u16(bytes, 22)? != 0 {
        return Err(ArtifactError::MalformedLayout {
            reason: "reserved header bits are nonzero",
        });
    }
    let section_count =
        usize::try_from(read_u32(bytes, 24)?).map_err(|_| ArtifactError::LengthOverflow)?;
    if section_count > MAX_SECTIONS {
        return Err(ArtifactError::TooManySections {
            actual: section_count,
            maximum: MAX_SECTIONS,
        });
    }
    let header_length =
        usize::try_from(read_u32(bytes, 28)?).map_err(|_| ArtifactError::LengthOverflow)?;
    let declared_total = read_u64(bytes, 32)?;
    if declared_total > MAX_ARTIFACT_LENGTH {
        return Err(ArtifactError::ArtifactTooLarge {
            actual: declared_total,
            maximum: MAX_ARTIFACT_LENGTH,
        });
    }
    if declared_total > actual_length {
        return Err(ArtifactError::Truncated {
            expected: declared_total,
            actual: actual_length,
        });
    }
    if declared_total < actual_length {
        return Err(ArtifactError::ExtraData {
            expected: declared_total,
            actual: actual_length,
        });
    }
    let table_length = section_count
        .checked_mul(SECTION_TABLE_ENTRY_LENGTH)
        .ok_or(ArtifactError::LengthOverflow)?;
    let expected_header_length = FIXED_HEADER_LENGTH
        .checked_add(target_length)
        .and_then(|length| length.checked_add(table_length))
        .and_then(|length| length.checked_add(HEADER_CHECKSUM_LENGTH))
        .ok_or(ArtifactError::LengthOverflow)?;
    if header_length != expected_header_length || header_length > bytes.len() - FOOTER_LENGTH {
        return Err(ArtifactError::MalformedLayout {
            reason: "header length does not match its target and section table",
        });
    }
    let header_checksum_offset = header_length - HEADER_CHECKSUM_LENGTH;
    let stored_header_checksum = read_u32(bytes, header_checksum_offset)?;
    if crc32fast::hash(&bytes[..header_checksum_offset]) != stored_header_checksum {
        return Err(ArtifactError::ChecksumMismatch {
            region: ChecksumRegion::Header,
        });
    }

    let project_fingerprint = array_16(bytes, 40)?;
    if project_fingerprint != expected_project_fingerprint {
        return Err(ArtifactError::ProjectFingerprintMismatch);
    }
    if array_16(bytes, 56)? != GENERATED_ENGINE_SEMANTICS_FINGERPRINT {
        return Err(ArtifactError::EngineFingerprintMismatch);
    }
    if encoded_endian != current_endian() {
        return Err(ArtifactError::EndianMismatch);
    }
    if encoded_pointer_width != current_pointer_width() {
        return Err(ArtifactError::PointerWidthMismatch);
    }
    let target_end = FIXED_HEADER_LENGTH
        .checked_add(target_length)
        .ok_or(ArtifactError::LengthOverflow)?;
    let encoded_target =
        bytes
            .get(FIXED_HEADER_LENGTH..target_end)
            .ok_or(ArtifactError::MalformedLayout {
                reason: "target marker lies outside the header",
            })?;
    if encoded_target != current_target().as_bytes() {
        return Err(ArtifactError::TargetMismatch {
            found: String::from_utf8_lossy(encoded_target).into_owned(),
            expected: current_target(),
        });
    }

    let footer_start = bytes.len() - FOOTER_LENGTH;
    if &bytes[footer_start..footer_start + COMMIT_MAGIC.len()] != COMMIT_MAGIC {
        return Err(ArtifactError::MissingCommitFooter);
    }
    let body_checksum = read_u32(bytes, footer_start + 16)?;
    if read_u64(bytes, footer_start + 20)? != declared_total {
        return Err(ArtifactError::MalformedLayout {
            reason: "commit footer total length disagrees with the header",
        });
    }
    let stored_footer_checksum = read_u32(bytes, footer_start + 28)?;
    if crc32fast::hash(&bytes[footer_start..footer_start + 28]) != stored_footer_checksum {
        return Err(ArtifactError::ChecksumMismatch {
            region: ChecksumRegion::Footer,
        });
    }
    if crc32fast::hash(&bytes[..footer_start]) != body_checksum {
        return Err(ArtifactError::ChecksumMismatch {
            region: ChecksumRegion::Body,
        });
    }

    let table_start = target_end;
    let mut encoded_sections = Vec::with_capacity(section_count);
    let mut seen_ids = BTreeSet::new();
    let mut expected_offset = header_length;
    for index in 0..section_count {
        let entry = table_start
            .checked_add(
                index
                    .checked_mul(SECTION_TABLE_ENTRY_LENGTH)
                    .ok_or(ArtifactError::LengthOverflow)?,
            )
            .ok_or(ArtifactError::LengthOverflow)?;
        let id = read_u32(bytes, entry)?;
        if !seen_ids.insert(id) {
            return Err(ArtifactError::DuplicateSection { id });
        }
        let offset = usize::try_from(read_u64(bytes, entry + 4)?)
            .map_err(|_| ArtifactError::LengthOverflow)?;
        let length_u64 = read_u64(bytes, entry + 12)?;
        if length_u64 > MAX_SECTION_LENGTH {
            return Err(ArtifactError::SectionTooLarge {
                id,
                actual: length_u64,
                maximum: MAX_SECTION_LENGTH,
            });
        }
        let length = usize::try_from(length_u64).map_err(|_| ArtifactError::LengthOverflow)?;
        if offset != expected_offset {
            return Err(ArtifactError::MalformedLayout {
                reason: "sections are not contiguous in table order",
            });
        }
        let end = offset
            .checked_add(length)
            .ok_or(ArtifactError::LengthOverflow)?;
        if end > footer_start {
            return Err(ArtifactError::MalformedLayout {
                reason: "section extends into the commit footer",
            });
        }
        encoded_sections.push(EncodedSection {
            id,
            offset,
            length,
            checksum: read_u32(bytes, entry + 20)?,
        });
        expected_offset = end;
    }
    if expected_offset != footer_start {
        return Err(ArtifactError::MalformedLayout {
            reason: "payload bytes are not fully described by the section table",
        });
    }

    for section in &encoded_sections {
        let payload = &bytes[section.offset..section.offset + section.length];
        if crc32fast::hash(payload) != section.checksum {
            return Err(ArtifactError::ChecksumMismatch {
                region: ChecksumRegion::Section(section.id),
            });
        }
    }
    Ok(ValidatedArtifact {
        project_fingerprint,
        sections: encoded_sections,
    })
}

fn validate_sections(sections: &[ArtifactSection]) -> Result<(), ArtifactError> {
    if sections.len() > MAX_SECTIONS {
        return Err(ArtifactError::TooManySections {
            actual: sections.len(),
            maximum: MAX_SECTIONS,
        });
    }
    let mut ids = BTreeSet::new();
    for section in sections {
        if !ids.insert(section.id) {
            return Err(ArtifactError::DuplicateSection { id: section.id });
        }
        let length =
            u64::try_from(section.payload().len()).map_err(|_| ArtifactError::LengthOverflow)?;
        if length > MAX_SECTION_LENGTH {
            return Err(ArtifactError::SectionTooLarge {
                id: section.id,
                actual: length,
                maximum: MAX_SECTION_LENGTH,
            });
        }
    }
    Ok(())
}

fn create_temporary_sibling(path: &Path) -> Result<(PathBuf, File), ArtifactError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| ArtifactError::Io {
        operation: "derive a temporary sibling for",
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::InvalidInput,
            "artifact path has no file name",
        ),
    })?;
    for _ in 0..128 {
        let sequence = TEMPORARY_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = file_name.to_os_string();
        temporary_name.push(format!(".tmp.{}.{sequence}", std::process::id()));
        let temporary_path = parent.join(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {}
            Err(source) => {
                return Err(ArtifactError::Io {
                    operation: "create temporary artifact beside",
                    path: path.to_path_buf(),
                    source,
                });
            }
        }
    }
    Err(ArtifactError::Io {
        operation: "create a unique temporary artifact beside",
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "temporary artifact name space exhausted",
        ),
    })
}

const fn current_target() -> &'static str {
    env!("DREAM64_ENGINE_TARGET")
}

const fn current_endian() -> u8 {
    if cfg!(target_endian = "little") { 1 } else { 2 }
}

const fn current_pointer_width() -> u8 {
    if cfg!(target_pointer_width = "64") {
        64
    } else if cfg!(target_pointer_width = "32") {
        32
    } else {
        16
    }
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ArtifactError> {
    let encoded = bytes
        .get(offset..offset + 2)
        .ok_or(ArtifactError::MalformedLayout {
            reason: "u16 field lies outside the artifact",
        })?;
    Ok(u16::from_le_bytes([encoded[0], encoded[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ArtifactError> {
    let encoded = bytes
        .get(offset..offset + 4)
        .ok_or(ArtifactError::MalformedLayout {
            reason: "u32 field lies outside the artifact",
        })?;
    Ok(u32::from_le_bytes([
        encoded[0], encoded[1], encoded[2], encoded[3],
    ]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, ArtifactError> {
    let encoded = bytes
        .get(offset..offset + 8)
        .ok_or(ArtifactError::MalformedLayout {
            reason: "u64 field lies outside the artifact",
        })?;
    Ok(u64::from_le_bytes([
        encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
        encoded[7],
    ]))
}

fn array_16(bytes: &[u8], offset: usize) -> Result<[u8; 16], ArtifactError> {
    bytes
        .get(offset..offset + 16)
        .ok_or(ArtifactError::MalformedLayout {
            reason: "fingerprint lies outside the artifact",
        })?
        .try_into()
        .map_err(|_| ArtifactError::MalformedLayout {
            reason: "fingerprint has the wrong length",
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    const PROJECT: [u8; 16] = [0x31; 16];
    const ARTIFACT_ABI_CRATES: [&str; 15] = [
        "dm-core",
        "dm-project",
        "dm-lexer",
        "dm-syntax",
        "dm-object-tree",
        "dm-compiler",
        "dm-lowering",
        "dm-semantics",
        "dm-value",
        "dm-vm",
        "dm-globals",
        "dm-runtime",
        "dm-map",
        "dm-world",
        "dm-lifecycle",
    ];
    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> CompiledArtifact {
        CompiledArtifact::new(
            PROJECT,
            vec![
                ArtifactSection::new(7, b"module bytecode".to_vec()),
                ArtifactSection::new(19, vec![0, 1, 2, 3, 255]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn deterministic_roundtrip_preserves_sections() {
        let artifact = fixture();
        let first = artifact.encode().unwrap();
        let second = artifact.encode().unwrap();
        assert_eq!(first, second);

        let decoded = CompiledArtifact::decode(&first, PROJECT).unwrap();
        assert_eq!(decoded, artifact);
        assert_eq!(decoded.encode().unwrap(), first);
        assert_eq!(decoded.section(7).unwrap().payload(), b"module bytecode");
    }

    #[test]
    fn engine_fingerprint_routes_every_artifact_abi_crate_and_build_input() {
        let build_script = include_str!("../build.rs");
        for crate_name in ARTIFACT_ABI_CRATES {
            assert!(
                build_script.contains(&format!("\"{crate_name}\"")),
                "engine fingerprint omits {crate_name}"
            );
        }
        for identity in [
            "dm-lifecycle/build.rs",
            "workspace/Cargo.toml",
            "workspace/Cargo.lock",
        ] {
            assert!(
                build_script.contains(identity),
                "engine fingerprint omits {identity}"
            );
        }
    }

    #[test]
    fn file_decode_shares_one_owned_backing_and_releases_it_with_sections() {
        let directory = TestDirectory::new();
        let path = directory.path.join("compiled.d64c");
        let encoded = fixture().encode().unwrap();
        fs::write(&path, &encoded).unwrap();

        let mut decoded = CompiledArtifact::read_from(&path, PROJECT).unwrap();
        let stats = decoded.storage_stats();
        assert_eq!(stats.payload_bytes, b"module bytecode".len() + 5);
        assert_eq!(stats.backing_allocations, 1);
        assert!(stats.backing_bytes >= encoded.len());

        let first_backing = match &decoded.section(7).unwrap().payload {
            ArtifactPayload::Shared { backing, .. } => Arc::downgrade(backing),
            ArtifactPayload::Owned(_) => panic!("file payload should share its input buffer"),
            ArtifactPayload::Mapped { .. } => panic!("small fixture should use buffered storage"),
        };
        let retained_section = decoded.sections.remove(0);
        assert_eq!(retained_section.payload(), b"module bytecode");
        drop(decoded);
        assert!(first_backing.upgrade().is_some());
        assert_eq!(retained_section.into_payload(), b"module bytecode");
        assert!(first_backing.upgrade().is_none());

        let mut corrupt = encoded;
        let payload_offset = usize::try_from(read_u32(&corrupt, 28).unwrap()).unwrap();
        corrupt[payload_offset] ^= 0x40;
        fs::write(&path, corrupt).unwrap();
        assert!(matches!(
            CompiledArtifact::read_from(&path, PROJECT),
            Err(ArtifactError::ChecksumMismatch {
                region: ChecksumRegion::Body,
            })
        ));
    }

    #[test]
    fn mapped_and_buffered_reads_are_equivalent_and_mapping_owns_sections() {
        let directory = TestDirectory::new();
        let path = directory.path.join("mapped.d64");
        fixture().write_atomic(&path).unwrap();
        let buffered =
            CompiledArtifact::read_from_with_mode(&path, PROJECT, ArtifactReadMode::Buffered)
                .unwrap();
        let mut mapped =
            CompiledArtifact::read_from_with_mode(&path, PROJECT, ArtifactReadMode::PreferMapped)
                .unwrap();
        assert_eq!(mapped, buffered);
        assert_eq!(mapped.storage_stats().mapped_backings, 1);
        assert_eq!(buffered.storage_stats().buffered_backings, 1);
        let owner = match &mapped.section(7).unwrap().payload {
            ArtifactPayload::Mapped { backing, .. } => Arc::downgrade(backing),
            _ => panic!("forced mapped read should retain a mapping"),
        };
        let section = mapped.sections.remove(0);
        drop(mapped);
        assert!(owner.upgrade().is_some());
        assert_eq!(section.payload(), b"module bytecode");
        drop(section);
        assert!(owner.upgrade().is_none());

        // Windows cannot replace/delete a file while a live mapping owns it.
        // Once the decoded artifact is dropped, normal atomic rebuild works.
        fixture().write_atomic(&path).unwrap();
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn mapping_failure_falls_back_and_mapped_corruption_is_rejected() {
        let directory = TestDirectory::new();
        let path = directory.path.join("fallback.d64");
        let mut encoded = fixture().encode().unwrap();
        fs::write(&path, &encoded).unwrap();
        let fallback = CompiledArtifact::read_from_with_mapper_for_test(
            &path,
            PROJECT,
            ArtifactReadMode::PreferMapped,
            |_| Err(io::Error::other("injected mapping failure")),
        )
        .unwrap();
        assert_eq!(fallback, fixture());
        assert_eq!(fallback.storage_stats().buffered_backings, 1);

        let payload_offset = usize::try_from(read_u32(&encoded, 28).unwrap()).unwrap();
        encoded[payload_offset] ^= 0x40;
        fs::write(&path, encoded).unwrap();
        assert!(matches!(
            CompiledArtifact::read_from_with_mode(&path, PROJECT, ArtifactReadMode::PreferMapped,),
            Err(ArtifactError::ChecksumMismatch {
                region: ChecksumRegion::Body,
            })
        ));
    }

    #[test]
    #[ignore = "manual artifact read microbenchmark"]
    fn benchmark_mmap_vs_buffered_artifact_load() {
        let directory = TestDirectory::new();
        let path = directory.path.join("benchmark.d64");
        let artifact = CompiledArtifact::new(
            PROJECT,
            vec![ArtifactSection::new(1, vec![0x5a; 32 * 1024 * 1024])],
        )
        .unwrap();
        artifact.write_atomic(&path).unwrap();
        for mode in [ArtifactReadMode::Buffered, ArtifactReadMode::PreferMapped] {
            let started = std::time::Instant::now();
            let mut checksum = 0usize;
            for _ in 0..10 {
                let loaded = CompiledArtifact::read_from_with_mode(&path, PROJECT, mode).unwrap();
                checksum ^= loaded.section(1).unwrap().payload()[0] as usize;
            }
            eprintln!(
                "artifact-read-benchmark mode={mode:?} iterations=10 bytes={} elapsed_ms={} checksum={checksum}",
                fs::metadata(&path).unwrap().len(),
                started.elapsed().as_millis(),
            );
        }
    }

    #[test]
    fn project_mismatch_is_rejected() {
        let encoded = fixture().encode().unwrap();
        assert!(matches!(
            CompiledArtifact::decode(&encoded, [0x92; 16]),
            Err(ArtifactError::ProjectFingerprintMismatch)
        ));
    }

    #[test]
    fn engine_mismatch_is_rejected() {
        let mut metadata = EncodingMetadata::current();
        metadata.engine_fingerprint[0] ^= 0xff;
        let encoded = fixture().encode_with_metadata(metadata).unwrap();
        assert!(matches!(
            CompiledArtifact::decode(&encoded, PROJECT),
            Err(ArtifactError::EngineFingerprintMismatch)
        ));
    }

    #[test]
    fn schema_mismatch_is_rejected() {
        let mut metadata = EncodingMetadata::current();
        metadata.schema = ARTIFACT_SCHEMA + 1;
        let encoded = fixture().encode_with_metadata(metadata).unwrap();
        assert!(matches!(
            CompiledArtifact::decode(&encoded, PROJECT),
            Err(ArtifactError::SchemaMismatch {
                found,
                expected: ARTIFACT_SCHEMA,
            }) if found == ARTIFACT_SCHEMA + 1
        ));
    }

    #[test]
    fn same_size_payload_corruption_is_rejected() {
        let mut encoded = fixture().encode().unwrap();
        let payload_offset = usize::try_from(read_u32(&encoded, 28).unwrap()).unwrap();
        encoded[payload_offset] ^= 0x40;
        assert!(matches!(
            CompiledArtifact::decode(&encoded, PROJECT),
            Err(ArtifactError::ChecksumMismatch {
                region: ChecksumRegion::Body,
            })
        ));
    }

    #[test]
    fn truncation_is_rejected() {
        let mut encoded = fixture().encode().unwrap();
        encoded.truncate(encoded.len() - 7);
        assert!(matches!(
            CompiledArtifact::decode(&encoded, PROJECT),
            Err(ArtifactError::Truncated { .. })
        ));
    }

    #[test]
    fn trailing_data_is_rejected() {
        let mut encoded = fixture().encode().unwrap();
        encoded.extend_from_slice(b"not part of artifact");
        assert!(matches!(
            CompiledArtifact::decode(&encoded, PROJECT),
            Err(ArtifactError::ExtraData { .. })
        ));
    }

    #[test]
    fn uncommitted_footer_is_rejected() {
        let mut encoded = fixture().encode().unwrap();
        let footer_start = encoded.len() - FOOTER_LENGTH;
        encoded[footer_start..footer_start + COMMIT_MAGIC.len()].fill(0);
        assert!(matches!(
            CompiledArtifact::decode(&encoded, PROJECT),
            Err(ArtifactError::MissingCommitFooter)
        ));
    }

    #[test]
    fn atomic_write_replaces_only_with_complete_artifact() {
        let directory = TestDirectory::new();
        let path = directory.path.join("compiled.d64c");
        let old = CompiledArtifact::new([0x11; 16], vec![ArtifactSection::new(1, b"old".to_vec())])
            .unwrap();
        old.write_atomic(&path).unwrap();
        assert_eq!(CompiledArtifact::read_from(&path, [0x11; 16]).unwrap(), old);

        let replacement = fixture();
        replacement.write_atomic(&path).unwrap();
        assert_eq!(
            CompiledArtifact::read_from(&path, PROJECT).unwrap(),
            replacement
        );
        let sibling_names = fs::read_dir(&directory.path)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert_eq!(sibling_names, [path.file_name().unwrap()]);
    }

    #[test]
    fn atomic_write_streams_sections_with_bounded_envelope_storage() {
        let directory = TestDirectory::new();
        let path = directory.path.join("large.d64c");
        let artifact = CompiledArtifact::new(
            PROJECT,
            vec![
                ArtifactSection::new(1, vec![0x5a; 1024 * 1024]),
                ArtifactSection::new(2, vec![0xa5; 1024 * 1024]),
            ],
        )
        .unwrap();
        let expected = artifact.encode().unwrap();

        let stats = artifact.write_atomic_with_stats(&path).unwrap();

        assert_eq!(stats.encoded_bytes, expected.len() as u64);
        assert_eq!(stats.payload_bytes, 2 * 1024 * 1024);
        assert_eq!(stats.write_calls, 4);
        assert!(
            stats.peak_staging_bytes < 4096,
            "the writer must stage only the small envelope, not either payload"
        );
        assert_eq!(fs::read(&path).unwrap(), expected);
        assert_eq!(
            CompiledArtifact::read_from(&path, PROJECT).unwrap(),
            artifact
        );
    }

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "dream64-artifact-test-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
