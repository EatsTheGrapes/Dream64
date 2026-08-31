//! Decoding and structural validation of encoded compiled artifacts: parsing
//! the fixed header, verifying every CRC32 region and cross-field invariant,
//! and rebuilding the section table into an owned, shared, or memory-mapped
//! `CompiledArtifact`.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io;
use std::path::Path;
use std::sync::Arc;

use super::{
    ARTIFACT_SCHEMA, ArtifactError, ArtifactReadMode, ArtifactSection, COMMIT_MAGIC,
    ChecksumRegion, CompiledArtifact, DEFAULT_MMAP_THRESHOLD, FIXED_HEADER_LENGTH, FOOTER_LENGTH,
    GENERATED_ENGINE_SEMANTICS_FINGERPRINT, HEADER_CHECKSUM_LENGTH, MAGIC, MAX_ARTIFACT_LENGTH,
    MAX_SECTION_LENGTH, MAX_SECTIONS, MAX_TARGET_LENGTH, SECTION_TABLE_ENTRY_LENGTH, array_16,
    current_endian, current_pointer_width, current_target, read_u16, read_u32, read_u64,
};

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
pub(super) fn decode_artifact(
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

pub(super) fn read_from_with_mapper(
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
