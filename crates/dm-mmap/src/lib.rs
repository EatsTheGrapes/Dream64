//! Narrow read-only memory-mapping boundary for immutable Dream64 artifacts.
//!
//! All callers remain safe Rust. The sole unsafe operation is the OS mapping
//! constructor required by `memmap2`; mapped bytes are never exposed mutably.

#![deny(missing_docs)]

use std::fs::File;
use std::io;
use std::ops::Deref;

/// An owning, immutable view of a file mapping.
pub struct ReadOnlyMapping(memmap2::Mmap);

impl ReadOnlyMapping {
    /// Maps the file read-only and retains the mapping owner until drop.
    ///
    /// # Errors
    ///
    /// Returns the platform mapping error. Callers should retain a buffered
    /// fallback because mapping can fail under address-space or policy limits.
    pub fn map(file: &File) -> io::Result<Self> {
        // SAFETY: `MmapOptions::map` creates a read-only view. This wrapper
        // exposes only `&[u8]`, owns the mapping for the full slice lifetime,
        // and provides no mutation API. Artifact files are installed by atomic
        // replacement, never modified in place while mapped.
        unsafe { memmap2::MmapOptions::new().map(file).map(Self) }
    }
}

impl Deref for ReadOnlyMapping {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[u8]> for ReadOnlyMapping {
    fn as_ref(&self) -> &[u8] {
        self
    }
}
