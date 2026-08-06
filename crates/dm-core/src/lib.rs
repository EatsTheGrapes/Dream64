//! Shared, representation-level contracts for the DM compiler and runtime.

#![cfg_attr(not(test), deny(missing_docs))]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("dm64 requires a 64-bit target");

use std::fmt;

/// Stable source-file identity within one loaded project.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FileId(u32);

impl FileId {
    /// Creates an identity from a project-local file index.
    ///
    /// # Panics
    ///
    /// Panics when `index` cannot be represented by the 32-bit project file
    /// table. This is an engine limit, not a process address-space limit.
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("a project cannot contain more than u32::MAX files"))
    }

    /// Returns the index used by project tables.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A byte offset range in one source file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceSpan {
    /// Inclusive byte offset at which the token starts.
    pub start: usize,
    /// Exclusive byte offset at which the token ends.
    pub end: usize,
}

impl SourceSpan {
    /// Creates a span and rejects inverted bounds.
    ///
    /// # Panics
    ///
    /// Panics when `end` is less than `start`.
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        assert!(start <= end, "a source span cannot end before it starts");
        Self { start, end }
    }

    /// Returns the length of the span in bytes.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end - self.start
    }

    /// Returns whether the span contains no bytes.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// The exact binary32 storage used by a compatibility-mode DM `num`.
///
/// Bits are retained instead of deriving `Eq` or `Hash` on `f32`. Runtime
/// equality and ordering are semantic operations and must not be confused with
/// storage identity, especially for signed zero and NaN payloads.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct DmNumberBits(u32);

impl DmNumberBits {
    /// Stores a host float without widening it.
    #[must_use]
    pub const fn from_f32(value: f32) -> Self {
        Self(value.to_bits())
    }

    /// Restores the stored binary32 value.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }

    /// Returns the unmodified IEEE-754 bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for DmNumberBits {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DmNumberBits")
            .field("value", &self.to_f32())
            .field("bits", &format_args!("{:#010x}", self.bits()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{DmNumberBits, SourceSpan};

    #[test]
    fn number_storage_preserves_signed_zero() {
        let positive = DmNumberBits::from_f32(0.0);
        let negative = DmNumberBits::from_f32(-0.0);

        assert_ne!(positive, negative);
        assert_eq!(positive.bits(), 0);
        assert_eq!(negative.bits(), 0x8000_0000);
    }

    #[test]
    fn source_span_reports_byte_length() {
        let span = SourceSpan::new(4, 9);

        assert_eq!(span.len(), 5);
        assert!(!span.is_empty());
    }
}
