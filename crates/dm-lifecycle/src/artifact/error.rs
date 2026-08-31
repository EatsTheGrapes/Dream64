//! Rejection reasons for encoded compiled artifacts.

use std::fmt;
use std::io;
use std::path::PathBuf;

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
