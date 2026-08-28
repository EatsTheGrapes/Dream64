//! Portable lifecycle/catalog products.
//!
//! This module is the intended home for artifact-time catalog data and its
//! serialization. Keep filesystem discovery, lifecycle execution, and IPC out
//! of this module.

use std::collections::BTreeMap;

/// A narrow catalog boundary for portable key/value products.
///
/// Concrete catalog types remain in the lifecycle crate while the migration is
/// in progress so the public API can remain stable.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Catalog<T>(pub BTreeMap<String, T>);

impl<T> Catalog<T> {
    /// Creates an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Returns the number of entries in the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the catalog has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
