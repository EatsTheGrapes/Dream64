//! Durable values, generational heap identities, and ordered DM list storage.
//!
//! This layer deliberately uses logical numeric handles instead of `Rc`/`Arc`
//! addresses for datum and list identity. A future serializer can remap those
//! handles without exposing process pointers. Text may share immutable storage,
//! but text equality is always content equality.
//!
//! Confirmed contracts represented here include binary32 numbers, 1-based list
//! positions, reference identity for lists/datums, and shallow list copies.
//! Cross-type coercive equality, sparse positional assignment, duplicate-value
//! removal breadth, and numeric assignment through an associative key's
//! iteration position still require differential BYOND fixtures.

#![cfg_attr(not(test), deny(missing_docs))]

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::BuildHasherDefault;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use ahash::AHashMap;
use dm_core::DmNumberBits;
use rayon::prelude::*;

mod snapshot;

/// A canonical absolute DM type path.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TypePath(Arc<str>);

impl TypePath {
    /// Validates and stores an absolute slash-delimited path.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::InvalidTypePath`] for relative paths, the root
    /// path, empty segments, or a trailing slash.
    pub fn parse(path: &str) -> Result<Self, ValueError> {
        if !path.starts_with('/')
            || path.len() == 1
            || path.ends_with('/')
            || path[1..].split('/').any(str::is_empty)
        {
            return Err(ValueError::InvalidTypePath(path.to_owned()));
        }
        Ok(Self(Arc::from(path)))
    }

    /// Returns the canonical path spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the process-local identity of this shared path allocation.
    ///
    /// This is an engine cache key only. Callers must retain a clone of the
    /// path for as long as the key is stored so allocator address reuse cannot
    /// alias an unrelated path.
    #[doc(hidden)]
    #[must_use]
    pub fn storage_identity(&self) -> usize {
        self.0.as_ptr() as usize
    }
}

impl Borrow<str> for TypePath {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for TypePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A canonical case-sensitive DM field identifier.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FieldName(Arc<str>);

impl FieldName {
    /// Produces a collision-free VM storage key for a canonical type-static
    /// variable path. Each source byte is hex encoded so owner qualification
    /// and reopening identity survive the identifier-only VM namespace.
    #[must_use]
    pub fn static_storage(variable_path: &str) -> Self {
        let mut encoded = String::from("__dm_static_");
        for byte in variable_path.bytes() {
            use std::fmt::Write as _;
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        Self(Arc::from(encoded))
    }

    /// Validates and stores one DM identifier.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::InvalidFieldName`] when `name` is empty, starts
    /// with a non-identifier character, or contains non-identifier bytes.
    pub fn parse(name: &str) -> Result<Self, ValueError> {
        let mut bytes = name.bytes();
        let Some(first) = bytes.next() else {
            return Err(ValueError::InvalidFieldName(name.to_owned()));
        };
        if !(first.is_ascii_alphabetic() || first == b'_')
            || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(ValueError::InvalidFieldName(name.to_owned()));
        }
        Ok(Self(Arc::from(name)))
    }

    /// Returns the canonical identifier spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FieldName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

macro_rules! handle_type {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name {
            index: u32,
            generation: u32,
        }

        impl $name {
            /// Reconstructs a stable handle from a persisted arena slot and generation.
            #[doc(hidden)]
            #[must_use]
            pub const fn from_parts(index: u32, generation: u32) -> Self {
                Self { index, generation }
            }

            /// Returns the stable slot index used by heap tables.
            #[must_use]
            pub const fn index(self) -> u32 {
                self.index
            }

            /// Returns the generation used to reject stale references.
            #[must_use]
            pub const fn generation(self) -> u32 {
                self.generation
            }
        }
    };
}

handle_type!(DatumId, "A generational identity for one heap datum.");
handle_type!(ListId, "A generational identity for one mutable heap list.");

/// A runtime DM value.
#[derive(Clone, Debug)]
pub enum Value {
    /// DM `null`.
    Null,
    /// Compatibility binary32 `num` storage.
    Number(DmNumberBits),
    /// Immutable text with content-based semantics.
    Text(Arc<str>),
    /// A BYOND file/resource value carrying its project-relative path.
    ///
    /// File values are intentionally distinct from text: `file("x")` and
    /// resource literals satisfy `isfile()`, while an ordinary string that
    /// happens to contain the same spelling does not.
    File(Arc<str>),
    /// A canonical type path value.
    TypePath(TypePath),
    /// A type path carrying evaluated per-construction field overrides.
    ModifiedTypePath(Arc<ModifiedTypePath>),
    /// Reference to a heap datum.
    Datum(DatumId),
    /// Reference to a mutable heap list.
    List(ListId),
}

const PACKED_TAG_SHIFT: u32 = 62;
const PACKED_PAYLOAD_MASK: u64 = (1_u64 << PACKED_TAG_SHIFT) - 1;
const PACKED_NULL: u64 = 0;
const PACKED_NUMBER: u64 = 1;
const PACKED_DATUM: u64 = 2;
const PACKED_LIST: u64 = 3;
const PACKED_HANDLE_COMPONENT_MASK: u32 = (1_u32 << 31) - 1;

/// Pointer-free runtime value used by bounded hot execution paths.
///
/// This first representation phase deliberately supports only values that fit
/// without an intern table. Text, files, and paths remain rich values until a
/// lifetime-safe runtime string/type pool is available.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedValue {
    bits: u64,
}

impl PackedValue {
    /// Creates packed DM null without crossing the rich-value boundary.
    #[must_use]
    pub const fn null() -> Self {
        Self { bits: 0 }
    }

    /// Creates a packed compatibility number without crossing the rich-value boundary.
    #[must_use]
    pub const fn number_bits(number: DmNumberBits) -> Self {
        Self {
            bits: (PACKED_NUMBER << PACKED_TAG_SHIFT) | number.bits() as u64,
        }
    }

    /// Creates a packed compatibility number without crossing the rich-value boundary.
    #[must_use]
    pub const fn number(value: f32) -> Self {
        Self::number_bits(DmNumberBits::from_f32(value))
    }

    /// Packs a value when it has a pointer-free representation.
    #[must_use]
    pub const fn try_from_value(value: &Value) -> Option<Self> {
        let (tag, payload) = match value {
            Value::Null => (PACKED_NULL, 0),
            Value::Number(number) => (PACKED_NUMBER, number.bits() as u64),
            Value::Datum(datum) => {
                if datum.index() > PACKED_HANDLE_COMPONENT_MASK
                    || datum.generation() > PACKED_HANDLE_COMPONENT_MASK
                {
                    return None;
                }
                (
                    PACKED_DATUM,
                    datum.index() as u64 | ((datum.generation() as u64) << 31),
                )
            }
            Value::List(list) => {
                if list.index() > PACKED_HANDLE_COMPONENT_MASK
                    || list.generation() > PACKED_HANDLE_COMPONENT_MASK
                {
                    return None;
                }
                (
                    PACKED_LIST,
                    list.index() as u64 | ((list.generation() as u64) << 31),
                )
            }
            Value::Text(_) | Value::File(_) | Value::TypePath(_) | Value::ModifiedTypePath(_) => {
                return None;
            }
        };
        Some(Self {
            bits: (tag << PACKED_TAG_SHIFT) | payload,
        })
    }

    /// Restores the exact rich value represented by this record.
    #[must_use]
    pub fn into_value(self) -> Value {
        let tag = self.bits >> PACKED_TAG_SHIFT;
        let payload = self.bits & PACKED_PAYLOAD_MASK;
        match tag {
            PACKED_NULL => Value::Null,
            PACKED_NUMBER => Value::Number(DmNumberBits::from_f32(f32::from_bits(payload as u32))),
            PACKED_DATUM => Value::Datum(DatumId::from_parts(
                payload as u32 & PACKED_HANDLE_COMPONENT_MASK,
                (payload >> 31) as u32,
            )),
            PACKED_LIST => Value::List(ListId::from_parts(
                payload as u32 & PACKED_HANDLE_COMPONENT_MASK,
                (payload >> 31) as u32,
            )),
            _ => unreachable!("PackedValue tags are private and constructor-validated"),
        }
    }

    /// Returns DM's numeric value for null/number arithmetic.
    #[must_use]
    pub fn as_number_or_null(self) -> Option<f32> {
        match self.bits >> PACKED_TAG_SHIFT {
            PACKED_NULL => Some(0.0),
            PACKED_NUMBER => Some(f32::from_bits(self.bits as u32)),
            _ => None,
        }
    }

    /// Returns the heap identity carried by this record, when any.
    #[must_use]
    pub fn heap_handles(self) -> (Option<DatumId>, Option<ListId>) {
        match self.into_value() {
            Value::Datum(datum) => (Some(datum), None),
            Value::List(list) => (None, Some(list)),
            _ => (None, None),
        }
    }
}

impl Value {
    /// Creates a compatibility number without widening it.
    #[must_use]
    pub const fn number(value: f32) -> Self {
        Self::Number(DmNumberBits::from_f32(value))
    }

    /// Creates immutable text.
    #[must_use]
    pub fn text(value: impl Into<Arc<str>>) -> Self {
        Self::Text(value.into())
    }

    /// Creates a first-class BYOND file/resource value.
    #[must_use]
    pub fn file(value: impl Into<Arc<str>>) -> Self {
        Self::File(value.into())
    }

    /// Returns the stored binary32 number when this value is numeric.
    #[must_use]
    pub const fn as_number(&self) -> Option<f32> {
        match self {
            Self::Number(number) => Some(number.to_f32()),
            Self::Null
            | Self::Text(_)
            | Self::File(_)
            | Self::TypePath(_)
            | Self::ModifiedTypePath(_)
            | Self::Datum(_)
            | Self::List(_) => None,
        }
    }

    /// Compares values using the current non-coercive DM equality contract.
    ///
    /// Numbers compare numerically, making signed zero equal and every NaN
    /// unequal. Text and type paths compare by content; heap values compare by
    /// logical handle identity. Cross-type coercions remain intentionally
    /// unsupported pending conformance evidence.
    #[must_use]
    pub fn semantic_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Number(left), Self::Number(right)) => {
                left.to_f32().partial_cmp(&right.to_f32()) == Some(std::cmp::Ordering::Equal)
            }
            (Self::Text(left), Self::Text(right)) => left == right,
            (Self::File(left), Self::File(right)) => left == right,
            (Self::TypePath(left), Self::TypePath(right)) => left == right,
            (Self::ModifiedTypePath(left), Self::ModifiedTypePath(right)) => left == right,
            (Self::Datum(left), Self::Datum(right)) => left == right,
            (Self::List(left), Self::List(right)) => left == right,
            _ => false,
        }
    }
}

impl ValueHeap {
    /// Canonicalizes stale heap references to `null`.
    ///
    /// Heap handles must remain stable internally for identity, but DM-visible
    /// semantics must treat deleted datums and lists as null-like values. This
    /// helper preserves internals while converting only externally observed liveness
    /// failures to `Value::Null`.
    pub fn canonicalize_value(&self, value: &Value) -> Value {
        match value {
            Value::Datum(datum) => self
                .datum(*datum)
                .is_ok()
                .then_some(value.clone())
                .unwrap_or(Value::Null),
            Value::List(list) => self
                .list(*list)
                .is_ok()
                .then_some(value.clone())
                .unwrap_or(Value::Null),
            _ => value.clone(),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.semantic_eq(other)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => formatter.write_str("null"),
            Self::Number(number) => write!(formatter, "{}", number.to_f32()),
            Self::Text(text) => write!(formatter, "{text:?}"),
            Self::File(path) => write!(formatter, "file({path:?})"),
            Self::TypePath(path) => write!(formatter, "{path}"),
            Self::ModifiedTypePath(path) => write!(formatter, "{}{{...}}", path.base),
            Self::Datum(id) => write!(formatter, "datum({id:?})"),
            Self::List(id) => write!(formatter, "list({id:?})"),
        }
    }
}

/// A first-class modified type path such as `/obj/item{amount = 15}`.
#[derive(Clone, Debug, PartialEq)]
pub struct ModifiedTypePath {
    base: TypePath,
    overrides: Vec<(FieldName, Value)>,
}

impl ModifiedTypePath {
    /// Creates a modified path from its canonical base and evaluated overrides.
    #[must_use]
    pub const fn new(base: TypePath, overrides: Vec<(FieldName, Value)>) -> Self {
        Self { base, overrides }
    }

    /// Returns the canonical base type.
    #[must_use]
    pub const fn base(&self) -> &TypePath {
        &self.base
    }

    /// Returns overrides in source order.
    #[must_use]
    pub fn overrides(&self) -> &[(FieldName, Value)] {
        &self.overrides
    }
}

/// One deterministic layer of type-level field defaults.
///
/// A frontend may resolve inheritance into an ancestor-to-descendant sequence
/// of these path-keyed layers without exposing compiler-specific node IDs.
#[derive(Clone, Debug, PartialEq)]
pub struct DatumDefaults {
    type_path: TypePath,
    fields: Vec<(FieldName, Value)>,
}

impl DatumDefaults {
    /// Creates an empty default layer declared by `type_path`.
    #[must_use]
    pub const fn new(type_path: TypePath) -> Self {
        Self {
            type_path,
            fields: Vec::new(),
        }
    }

    /// Returns the type that declared this layer.
    #[must_use]
    pub const fn type_path(&self) -> &TypePath {
        &self.type_path
    }

    /// Inserts or updates a default while retaining first-declaration order.
    pub fn set(&mut self, name: FieldName, value: Value) -> Option<Value> {
        set_named_field(&mut self.fields, name, value)
    }

    /// Iterates defaults in deterministic declaration order.
    #[must_use]
    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&FieldName, &Value)> {
        self.fields.iter().map(|(name, value)| (name, value))
    }
}

// Small datums stay on the compact linear path. Larger datums pay for one
// shared hash table in exchange for constant-time hot field access.
const DATUM_FIELD_INDEX_THRESHOLD: usize = 16;

// GC capacity reclamation requires meaningful absolute waste as well as a 50%
// oversize ratio. Small vectors shrink exactly because Boot203 proved their
// aggregate slack dominates; page-scale vectors retain growth headroom.
const GC_VECTOR_SHRINK_MIN_WASTED_BYTES: usize = 64;
const GC_SHRINK_RATIO_NUMERATOR: usize = 3;
const GC_SHRINK_RATIO_DENOMINATOR: usize = 2;
const GC_VECTOR_SHRINK_HEADROOM_DENOMINATOR: usize = 4;
const GC_VECTOR_SMALL_ALLOCATION_BYTES: usize = 4 * 1_024;
// Boot204 retained 7.32m spare datum-field-index slots and 2.90m spare list
// association-index slots, but the slack was spread across hundreds of
// thousands of indexes. A 128-slot per-index floor therefore reclaimed none
// of it. Eight slots still represents at least 192 bytes of entry storage for
// the smaller field index while allowing the hash table's bucket granularity
// to decide whether a lower allocation can retain 25% growth headroom.
const GC_INDEX_SHRINK_MIN_EXCESS_ENTRIES: usize = 8;
const GC_INDEX_SHRINK_HEADROOM_DENOMINATOR: usize = 16;

fn gc_vector_shrink_target<T>(values: &Vec<T>) -> Option<usize> {
    let excess = values.capacity().saturating_sub(values.len());
    let excess_bytes = excess.saturating_mul(std::mem::size_of::<T>());
    let grossly_oversized = values
        .capacity()
        .saturating_mul(GC_SHRINK_RATIO_DENOMINATOR)
        >= values.len().saturating_mul(GC_SHRINK_RATIO_NUMERATOR);
    if !grossly_oversized || excess_bytes < GC_VECTOR_SHRINK_MIN_WASTED_BYTES {
        return None;
    }
    // Boot203's average slack was only ~142 bytes per allocated list, so a
    // page-scale floor alone misses the dominant millions-small-vectors case.
    // Small allocations are likely stable metadata lists and shrink exactly;
    // page-scale vectors retain 25% capacity for continued startup growth.
    let allocation_bytes = values.capacity().saturating_mul(std::mem::size_of::<T>());
    let target = if allocation_bytes < GC_VECTOR_SMALL_ALLOCATION_BYTES {
        values.len()
    } else {
        let headroom = values
            .len()
            .saturating_add(GC_VECTOR_SHRINK_HEADROOM_DENOMINATOR - 1)
            / GC_VECTOR_SHRINK_HEADROOM_DENOMINATOR;
        values.len().saturating_add(headroom)
    };
    (target < values.capacity()).then_some(target)
}

fn gc_shrink_vector<T>(values: &mut Vec<T>) -> usize {
    let Some(target) = gc_vector_shrink_target(values) else {
        return 0;
    };
    let before = values.capacity();
    values.shrink_to(target);
    before
        .saturating_sub(values.capacity())
        .saturating_mul(std::mem::size_of::<T>())
}

fn gc_index_shrink_target(len: usize, capacity: usize) -> Option<usize> {
    let excess = capacity.saturating_sub(len);
    if excess < GC_INDEX_SHRINK_MIN_EXCESS_ENTRIES {
        return None;
    }
    // Standard HashMap capacities move in coarse bucket classes. Boot205's
    // retained indexes naturally sat below the old 1.5x ratio gate, and a 25%
    // requested headroom still selected their current bucket class, producing
    // zero successful shrinks. Ask for the next lower class only when it can
    // retain at least 6.25% immediate growth; HashMap::shrink_to remains the
    // final bucket-aware authority and safely no-ops otherwise.
    let headroom = len.saturating_add(GC_INDEX_SHRINK_HEADROOM_DENOMINATOR - 1)
        / GC_INDEX_SHRINK_HEADROOM_DENOMINATOR;
    let target = len.saturating_add(headroom);
    (target < capacity).then_some(target)
}

fn gc_index_ratio_bin(len: usize, capacity: usize) -> usize {
    if capacity.saturating_mul(8) < len.saturating_mul(9) {
        0
    } else if capacity.saturating_mul(4) < len.saturating_mul(5) {
        1
    } else if capacity.saturating_mul(2) < len.saturating_mul(3) {
        2
    } else if capacity < len.saturating_mul(2) {
        3
    } else {
        4
    }
}

// Field lookup is the hottest generic operation during map and atom startup.
// These maps are process-local derived indexes, so cryptographic SipHash does
// not buy observable DM semantics or persistence safety. AHash retains keyed
// collision resistance while substantially reducing millions of short
// identifier lookups across mapping, atoms, lighting, atmos, and shuttles.
type DatumFieldIndex = AHashMap<FieldName, usize>;

const DATUM_LAYOUT_CACHE_MAX_ENTRIES: usize = 65_536;
const DATUM_LAYOUT_CACHE_MAX_VARIANTS_PER_TYPE: usize = 8;

#[derive(Clone)]
struct CachedDatumLayout {
    names: Arc<[FieldName]>,
    field_index: Option<Arc<DatumFieldIndex>>,
}

/// Retains the stable field shape shared by newly initialized datums.
///
/// GC-time interning is still required for shapes produced by arbitrary DM
/// mutation, but waiting for GC leaves every startup datum's wide
/// `(FieldName, Value)` vector live at once. This bounded cache moves the
/// compaction boundary to the end of instance initialization so those wide
/// temporary vectors can be reused immediately instead of defining the
/// process peak.
#[derive(Default)]
struct DatumLayoutCache {
    by_type: HashMap<TypePath, Vec<CachedDatumLayout>>,
    entries: usize,
}

impl DatumLayoutCache {
    fn compact(&mut self, datum: &mut Datum) {
        if datum.fields.is_empty() || matches!(datum.fields, DatumFields::Shared { .. }) {
            return;
        }

        if let Some(layouts) = self.by_type.get(datum.type_path())
            && let Some(layout) = layouts.iter().find(|layout| {
                layout.names.len() == datum.fields.len()
                    && layout
                        .names
                        .iter()
                        .zip(datum.fields.names())
                        .all(|(left, right)| left == right)
            })
        {
            datum.field_index = layout.field_index.clone();
            datum.fields.share_names(Arc::clone(&layout.names));
            return;
        }

        let names: Arc<[FieldName]> = datum.fields.names().cloned().collect();
        let layout = CachedDatumLayout {
            names: Arc::clone(&names),
            field_index: datum.field_index.clone(),
        };
        datum.fields.share_names(names);

        let layouts = self.by_type.entry(datum.type_path.clone()).or_default();
        if self.entries < DATUM_LAYOUT_CACHE_MAX_ENTRIES
            && layouts.len() < DATUM_LAYOUT_CACHE_MAX_VARIANTS_PER_TYPE
        {
            layouts.push(layout);
            self.entries += 1;
        }
    }
}

#[derive(Default)]
struct FieldIndexInterner {
    by_layout: HashMap<(u64, u64), Arc<DatumFieldIndex>>,
    by_pointer: HashMap<usize, Arc<DatumFieldIndex>>,
    names_by_index: HashMap<usize, Arc<[FieldName]>>,
    unindexed_names_by_layout: HashMap<(u64, u64), Arc<[FieldName]>>,
    unindexed_names_by_pointer: HashMap<usize, Arc<[FieldName]>>,
}

impl FieldIndexInterner {
    fn layout_fingerprint(fields: &DatumFields) -> (u64, u64) {
        let mut first = DefaultHasher::new();
        let mut second = DefaultHasher::new();
        fields.len().hash(&mut first);
        0x9e37_79b9_7f4a_7c15u64.hash(&mut second);
        fields.len().hash(&mut second);
        for (position, name) in fields.names().enumerate() {
            name.hash(&mut first);
            position.hash(&mut second);
            name.hash(&mut second);
        }
        (first.finish(), second.finish())
    }

    fn matches_layout(index: &DatumFieldIndex, fields: &DatumFields) -> bool {
        index.len() == fields.len()
            && fields
                .names()
                .enumerate()
                .all(|(position, name)| index.get(name) == Some(&position))
    }

    fn redirect(
        index: &mut Arc<DatumFieldIndex>,
        existing: &Arc<DatumFieldIndex>,
        aggregate: &mut DatumStorageStats,
    ) {
        if Arc::ptr_eq(existing, index) {
            return;
        }
        let old_capacity = index.capacity();
        let releases_allocation = Arc::strong_count(index) == 1;
        *index = Arc::clone(existing);
        aggregate.deduplicated_field_indexes =
            aggregate.deduplicated_field_indexes.saturating_add(1);
        if releases_allocation {
            aggregate.deduplicated_field_index_bytes =
                aggregate.deduplicated_field_index_bytes.saturating_add(
                    old_capacity.saturating_mul(std::mem::size_of::<(FieldName, usize)>()),
                );
        }
    }

    fn intern(
        &mut self,
        fields: &DatumFields,
        index: &mut Arc<DatumFieldIndex>,
        aggregate: &mut DatumStorageStats,
    ) -> (bool, Arc<[FieldName]>) {
        let source_pointer = Arc::as_ptr(index) as usize;
        if let Some(existing) = self.by_pointer.get(&source_pointer).cloned() {
            aggregate.field_index_pointer_cache_hits =
                aggregate.field_index_pointer_cache_hits.saturating_add(1);
            Self::redirect(index, &existing, aggregate);
            let names = self.canonical_names(fields, index);
            return (false, names);
        }

        aggregate.field_index_fingerprints_computed = aggregate
            .field_index_fingerprints_computed
            .saturating_add(1);
        let fingerprint = Self::layout_fingerprint(fields);
        if let Some(existing) = self.by_layout.get(&fingerprint) {
            // Two independent fingerprints plus an exact name/position check
            // make sharing collision-safe rather than hash-equality based.
            aggregate.field_index_exact_layout_comparisons = aggregate
                .field_index_exact_layout_comparisons
                .saturating_add(1);
            if Self::matches_layout(existing, fields) {
                let canonical = Arc::clone(existing);
                Self::redirect(index, &canonical, aggregate);
                self.by_pointer.insert(source_pointer, canonical);
                let names = self.canonical_names(fields, index);
                return (false, names);
            }
            // An exact fingerprint collision must never merge unlike layouts.
            // It is too rare to justify retaining an allocation-heavy collision
            // side table throughout a memory-pressure GC.
            aggregate.field_index_fingerprint_collisions = aggregate
                .field_index_fingerprint_collisions
                .saturating_add(1);
            self.by_pointer.insert(source_pointer, Arc::clone(index));
            let names = self.canonical_names(fields, index);
            return (true, names);
        }
        let canonical = Arc::clone(index);
        self.by_layout.insert(fingerprint, Arc::clone(&canonical));
        self.by_pointer.insert(source_pointer, canonical);
        let names = self.canonical_names(fields, index);
        (true, names)
    }

    fn canonical_names(
        &mut self,
        fields: &DatumFields,
        index: &Arc<DatumFieldIndex>,
    ) -> Arc<[FieldName]> {
        let pointer = Arc::as_ptr(index) as usize;
        if let Some(names) = self.names_by_index.get(&pointer) {
            return Arc::clone(names);
        }
        let names: Arc<[FieldName]> = match fields {
            DatumFields::Shared { names, .. } => Arc::clone(names),
            DatumFields::Owned(_) => fields.names().cloned().collect(),
        };
        self.names_by_index.insert(pointer, Arc::clone(&names));
        names
    }

    fn intern_unindexed_names(&mut self, fields: &DatumFields) -> Arc<[FieldName]> {
        if let DatumFields::Shared { names, .. } = fields {
            let pointer = Arc::as_ptr(names) as *const FieldName as usize;
            if let Some(existing) = self.unindexed_names_by_pointer.get(&pointer) {
                return Arc::clone(existing);
            }
        }
        let fingerprint = Self::layout_fingerprint(fields);
        if let Some(existing) = self.unindexed_names_by_layout.get(&fingerprint)
            && existing.len() == fields.len()
            && existing
                .iter()
                .zip(fields.names())
                .all(|(left, right)| left == right)
        {
            let existing = Arc::clone(existing);
            if let DatumFields::Shared { names, .. } = fields {
                let pointer = Arc::as_ptr(names) as *const FieldName as usize;
                self.unindexed_names_by_pointer
                    .insert(pointer, Arc::clone(&existing));
            }
            return existing;
        }
        let names: Arc<[FieldName]> = match fields {
            DatumFields::Shared { names, .. } => Arc::clone(names),
            DatumFields::Owned(_) => fields.names().cloned().collect(),
        };
        self.unindexed_names_by_layout
            .entry(fingerprint)
            .or_insert_with(|| Arc::clone(&names));
        let pointer = Arc::as_ptr(&names) as *const FieldName as usize;
        self.unindexed_names_by_pointer
            .insert(pointer, Arc::clone(&names));
        names
    }

    fn shared_name_layouts(&self) -> usize {
        self.names_by_index
            .len()
            .saturating_add(self.unindexed_names_by_layout.len())
    }

    fn shared_name_slots(&self) -> usize {
        self.names_by_index
            .values()
            .chain(self.unindexed_names_by_layout.values())
            .map(|names| names.len())
            .sum()
    }
}

/// Aggregate datum backing-storage telemetry collected during heap GC.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DatumStorageStats {
    /// Materialized datum field entries.
    pub field_len: usize,
    /// Allocated datum field-vector capacity.
    pub field_capacity: usize,
    /// Live datums using a shared immutable field-name layout.
    pub shared_field_name_datums: usize,
    /// Logical field-name slots represented by those shared layouts.
    pub shared_field_name_logical_slots: usize,
    /// Distinct immutable field-name layouts retained after interning.
    pub shared_field_name_layouts: usize,
    /// Physical field-name slots across the distinct shared layouts.
    pub shared_field_name_physical_slots: usize,
    /// Duplicate field-name handle bytes avoided by layout sharing.
    pub shared_field_name_bytes_saved: usize,
    /// Field vectors whose significant excess capacity was released.
    pub shrunk_field_vectors: usize,
    /// Field-vector capacity bytes returned to the allocator.
    pub reclaimed_capacity_bytes: usize,
    /// Retained adaptive field lookup indexes.
    pub field_indexes: usize,
    /// Entries stored across retained field lookup indexes.
    pub field_index_len: usize,
    /// Hash-table capacity across retained field lookup indexes.
    pub field_index_capacity: usize,
    /// Index counts in capacity/length bins: `<1.125`, `<1.25`, `<1.5`,
    /// `<2.0`, and `>=2.0`.
    pub field_index_ratio_bins: [usize; 5],
    /// Field lookup indexes whose excess bucket capacity was released.
    pub shrunk_field_indexes: usize,
    /// Hash-table capacity slots released from field lookup indexes.
    pub reclaimed_field_index_capacity: usize,
    /// Entry-storage bytes represented by released field-index capacity.
    ///
    /// This excludes the standard library hash table's private control-byte
    /// allocation, so it is a conservative measure of allocator bytes freed.
    pub reclaimed_field_index_bytes: usize,
    /// Datum index identities redirected to an existing identical field layout.
    pub deduplicated_field_indexes: usize,
    /// Entry-storage bytes released by field-layout index sharing.
    pub deduplicated_field_index_bytes: usize,
    /// Distinct physical field-index allocations retained after interning.
    pub physical_field_indexes: usize,
    /// Entries across distinct physical field-index allocations.
    pub physical_field_index_len: usize,
    /// Capacity across distinct physical field-index allocations.
    pub physical_field_index_capacity: usize,
    /// Unlike layouts that produced both identical 128-bit fingerprints.
    pub field_index_fingerprint_collisions: usize,
    /// Physical index layouts fingerprinted during this collection.
    pub field_index_fingerprints_computed: usize,
    /// Logical indexes resolved through an already-seen physical Arc pointer.
    pub field_index_pointer_cache_hits: usize,
    /// Full layout verifications performed for matching fingerprints.
    pub field_index_exact_layout_comparisons: usize,
}

/// One datum record retained by the heap.
#[derive(Clone)]
pub struct Datum {
    type_path: TypePath,
    fields: DatumFields,
    field_index: Option<Arc<DatumFieldIndex>>,
}

#[derive(Clone)]
enum DatumFields {
    Owned(Vec<(FieldName, Value)>),
    Shared {
        names: Arc<[FieldName]>,
        values: Vec<Value>,
    },
}

impl fmt::Debug for DatumFields {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}

impl PartialEq for DatumFields {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(left, right)| left == right)
    }
}

impl Default for DatumFields {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}

impl DatumFields {
    fn len(&self) -> usize {
        match self {
            Self::Owned(fields) => fields.len(),
            Self::Shared { values, .. } => values.len(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Owned(fields) => fields.capacity(),
            Self::Shared { values, .. } => values.capacity(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(test)]
    fn truncate(&mut self, len: usize) {
        self.materialize_owned().truncate(len);
    }

    fn value(&self, index: usize) -> &Value {
        match self {
            Self::Owned(fields) => &fields[index].1,
            Self::Shared { values, .. } => &values[index],
        }
    }

    fn name(&self, index: usize) -> &FieldName {
        match self {
            Self::Owned(fields) => &fields[index].0,
            Self::Shared { names, .. } => &names[index],
        }
    }

    fn value_mut(&mut self, index: usize) -> &mut Value {
        match self {
            Self::Owned(fields) => &mut fields[index].1,
            Self::Shared { values, .. } => &mut values[index],
        }
    }

    fn position(&self, name: &FieldName) -> Option<usize> {
        match self {
            Self::Owned(fields) => fields.iter().position(|(candidate, _)| candidate == name),
            Self::Shared { names, .. } => names.iter().position(|candidate| candidate == name),
        }
    }

    fn names(&self) -> impl ExactSizeIterator<Item = &FieldName> {
        enum Names<'a> {
            Owned(std::slice::Iter<'a, (FieldName, Value)>),
            Shared(std::slice::Iter<'a, FieldName>),
        }
        impl<'a> Iterator for Names<'a> {
            type Item = &'a FieldName;

            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    Self::Owned(fields) => fields.next().map(|(name, _)| name),
                    Self::Shared(names) => names.next(),
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                match self {
                    Self::Owned(fields) => fields.size_hint(),
                    Self::Shared(names) => names.size_hint(),
                }
            }
        }
        impl ExactSizeIterator for Names<'_> {}

        match self {
            Self::Owned(fields) => Names::Owned(fields.iter()),
            Self::Shared { names, .. } => Names::Shared(names.iter()),
        }
    }

    fn iter(&self) -> DatumFieldsIter<'_> {
        DatumFieldsIter {
            fields: self,
            index: 0,
        }
    }

    fn materialize_owned(&mut self) -> &mut Vec<(FieldName, Value)> {
        if let Self::Shared { names, values } = self {
            let fields = names.iter().cloned().zip(std::mem::take(values)).collect();
            *self = Self::Owned(fields);
        }
        let Self::Owned(fields) = self else {
            unreachable!("shared datum fields were materialized")
        };
        fields
    }

    fn share_names(&mut self, names: Arc<[FieldName]>) {
        if let Self::Shared { names: current, .. } = self {
            *current = names;
            return;
        }
        let Self::Owned(fields) = std::mem::take(self) else {
            unreachable!("owned datum fields were selected")
        };
        debug_assert_eq!(fields.len(), names.len());
        // Do not let Vec's in-place collect specialization reuse the wider
        // `(FieldName, Value)` allocation for the narrower value vector. That
        // would preserve the old byte capacity and defeat the compaction.
        let mut values = Vec::with_capacity(fields.len());
        values.extend(fields.into_iter().map(|(_, value)| value));
        *self = Self::Shared { names, values };
    }

    fn shrink_values_for_gc(&mut self) -> usize {
        match self {
            Self::Owned(fields) => gc_shrink_vector(fields),
            Self::Shared { values, .. } => gc_shrink_vector(values),
        }
    }
}

struct DatumFieldsIter<'a> {
    fields: &'a DatumFields,
    index: usize,
}

impl<'a> Iterator for DatumFieldsIter<'a> {
    type Item = (&'a FieldName, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        let item = match self.fields {
            DatumFields::Owned(fields) => fields.get(index).map(|(name, value)| (name, value)),
            DatumFields::Shared { names, values } => names.get(index).zip(values.get(index)),
        }?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.fields.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DatumFieldsIter<'_> {}

#[allow(clippy::missing_fields_in_debug)] // the cache is not logical datum state
impl fmt::Debug for Datum {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Datum")
            .field("type_path", &self.type_path)
            .field("fields", &self.fields)
            .finish()
    }
}

impl PartialEq for Datum {
    fn eq(&self, other: &Self) -> bool {
        self.type_path == other.type_path && self.fields == other.fields
    }
}

impl Datum {
    /// Returns the datum's canonical runtime type identity.
    #[must_use]
    pub const fn type_path(&self) -> &TypePath {
        &self.type_path
    }

    /// Returns the number of materialized named fields.
    #[must_use]
    pub fn field_len(&self) -> usize {
        self.fields.len()
    }

    /// Returns whether no named fields are materialized.
    #[must_use]
    pub fn fields_are_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Reads a named field.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::MissingField`] when `name` is not materialized.
    pub fn field(&self, name: &FieldName) -> Result<&Value, ValueError> {
        self.field_optional(name)
            .ok_or_else(|| ValueError::MissingField(name.clone()))
    }

    /// Reads a named field without allocating an error for an absent sparse slot.
    #[must_use]
    pub fn field_optional(&self, name: &FieldName) -> Option<&Value> {
        self.field_slot(name).map(|index| self.fields.value(index))
    }

    /// Returns the current positional slot for a materialized field.
    #[must_use]
    pub fn field_slot(&self, name: &FieldName) -> Option<usize> {
        if let Some(field_index) = &self.field_index {
            return field_index.get(name).copied();
        }
        self.fields.position(name)
    }

    /// Reads a cached positional slot only when it still names the expected field.
    ///
    /// Datum mutations can shift slots, so callers must retain and provide the
    /// stable field identity rather than trusting a stale numeric offset.
    #[must_use]
    pub fn field_at_validated_slot(&self, slot: usize, name: &FieldName) -> Option<&Value> {
        (slot < self.fields.len() && self.fields.name(slot) == name)
            .then(|| self.fields.value(slot))
    }

    /// Inserts or updates a field while retaining first-insertion order.
    pub fn set_field(&mut self, name: FieldName, value: Value) -> Option<Value> {
        if let Some(index) = self
            .field_index
            .as_ref()
            .and_then(|field_index| field_index.get(&name).copied())
        {
            return Some(std::mem::replace(self.fields.value_mut(index), value));
        }

        if let Some(field_index) = &mut self.field_index {
            let index = self.fields.len();
            self.fields.materialize_owned().push((name.clone(), value));
            let previous = Arc::make_mut(field_index).insert(name, index);
            debug_assert!(previous.is_none());
            return None;
        }

        let fields = self.fields.materialize_owned();
        let previous = set_named_field(fields, name, value);
        if previous.is_none() && self.fields.len() == DATUM_FIELD_INDEX_THRESHOLD {
            self.build_field_index();
        }
        previous
    }

    /// Deletes a materialized field and returns its value when present.
    pub fn delete_field(&mut self, name: &FieldName) -> Option<Value> {
        let index = match &mut self.field_index {
            Some(field_index) => {
                let index = *field_index.get(name)?;
                let removed = Arc::make_mut(field_index).remove(name);
                debug_assert_eq!(removed, Some(index));
                index
            }
            None => self.fields.position(name)?,
        };
        let value = self.fields.materialize_owned().remove(index).1;

        if self.fields.len() < DATUM_FIELD_INDEX_THRESHOLD {
            self.field_index = None;
        } else if let Some(field_index) = &mut self.field_index {
            let field_index = Arc::make_mut(field_index);
            for (offset, (shifted_name, _)) in
                self.fields.materialize_owned()[index..].iter().enumerate()
            {
                let _ = field_index.insert(shifted_name.clone(), index + offset);
            }
        }

        Some(value)
    }

    /// Applies one resolved type-default layer.
    ///
    /// Existing fields are updated in place and new fields append in layer
    /// order. Applying ancestor layers before descendants therefore preserves
    /// stable declaration order while allowing child overrides. Values are
    /// cloned shallowly, retaining heap-handle alias identity.
    pub fn apply_defaults(&mut self, defaults: &DatumDefaults) {
        for (name, value) in &defaults.fields {
            self.set_field(name.clone(), value.clone());
        }
    }

    /// Iterates materialized fields in stable first-declaration order.
    #[must_use]
    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&FieldName, &Value)> {
        self.fields.iter()
    }

    fn build_field_index(&mut self) {
        debug_assert!(self.fields.len() >= DATUM_FIELD_INDEX_THRESHOLD);
        self.field_index = Some(Arc::new(
            self.fields
                .names()
                .enumerate()
                .map(|(index, name)| (name.clone(), index))
                .collect(),
        ));
    }

    fn compact_and_measure_for_gc(
        &mut self,
        aggregate: &mut DatumStorageStats,
        field_indexes: &mut FieldIndexInterner,
    ) {
        let reclaimed = self.fields.shrink_values_for_gc();
        aggregate.shrunk_field_vectors = aggregate
            .shrunk_field_vectors
            .saturating_add(usize::from(reclaimed > 0));
        aggregate.reclaimed_capacity_bytes =
            aggregate.reclaimed_capacity_bytes.saturating_add(reclaimed);
        aggregate.field_len = aggregate.field_len.saturating_add(self.fields.len());
        aggregate.field_capacity = aggregate
            .field_capacity
            .saturating_add(self.fields.capacity());

        if let Some(index) = &mut self.field_index {
            aggregate.field_indexes = aggregate.field_indexes.saturating_add(1);
            let ratio_bin = gc_index_ratio_bin(index.len(), index.capacity());
            aggregate.field_index_ratio_bins[ratio_bin] =
                aggregate.field_index_ratio_bins[ratio_bin].saturating_add(1);
            if Arc::strong_count(index) == 1
                && let Some(target) = gc_index_shrink_target(index.len(), index.capacity())
            {
                let index = Arc::get_mut(index)
                    .expect("a uniquely owned datum field index must be mutable");
                let before = index.capacity();
                index.shrink_to(target);
                let reclaimed = before.saturating_sub(index.capacity());
                aggregate.shrunk_field_indexes = aggregate
                    .shrunk_field_indexes
                    .saturating_add(usize::from(reclaimed > 0));
                aggregate.reclaimed_field_index_capacity = aggregate
                    .reclaimed_field_index_capacity
                    .saturating_add(reclaimed);
                aggregate.reclaimed_field_index_bytes =
                    aggregate.reclaimed_field_index_bytes.saturating_add(
                        reclaimed.saturating_mul(std::mem::size_of::<(FieldName, usize)>()),
                    );
            }
            let (physical_is_new, names) = field_indexes.intern(&self.fields, index, aggregate);
            self.fields.share_names(names);
            aggregate.field_index_len = aggregate.field_index_len.saturating_add(index.len());
            aggregate.field_index_capacity = aggregate
                .field_index_capacity
                .saturating_add(index.capacity());
            if physical_is_new {
                aggregate.physical_field_indexes =
                    aggregate.physical_field_indexes.saturating_add(1);
                aggregate.physical_field_index_len = aggregate
                    .physical_field_index_len
                    .saturating_add(index.len());
                aggregate.physical_field_index_capacity = aggregate
                    .physical_field_index_capacity
                    .saturating_add(index.capacity());
            }
        } else if !self.fields.is_empty() {
            let names = field_indexes.intern_unindexed_names(&self.fields);
            self.fields.share_names(names);
        }
        if matches!(self.fields, DatumFields::Shared { .. }) {
            aggregate.shared_field_name_datums =
                aggregate.shared_field_name_datums.saturating_add(1);
            aggregate.shared_field_name_logical_slots = aggregate
                .shared_field_name_logical_slots
                .saturating_add(self.fields.len());
        }
    }
}

fn set_named_field(
    fields: &mut Vec<(FieldName, Value)>,
    name: FieldName,
    value: Value,
) -> Option<Value> {
    if let Some((_, current)) = fields.iter_mut().find(|(candidate, _)| candidate == &name) {
        return Some(std::mem::replace(current, value));
    }
    fields.push((name, value));
    None
}

/// Ordered mutable list data.
///
/// One insertion order covers positional values and associative keys. Numeric
/// indexing and iteration observe that unified order, while key lookup returns
/// the associated value. Associative reassignment preserves insertion order.
#[derive(Clone, Debug, Default)]
pub struct DmList {
    storage: Option<Arc<DmListStorage>>,
}

/// Lazily allocated storage for a non-empty or previously mutated DM list.
///
/// This type is public only because [`DmList`] uses standard dereferencing to
/// keep its implementation compact. Its fields remain private; callers should
/// continue to use the `DmList` API.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct DmListStorage {
    positional: Vec<Value>,
    associative: Vec<(Value, Value)>,
    order: Vec<ListOrder>,
    associative_index: Option<Box<AssociativeIndex>>,
    positional_remove_index: Option<Box<PositionalRemoveIndex>>,
    /// Logically removed prefix for canonical purely positional lists. This
    /// makes repeated `Cut(1, n)` queue drains amortized linear.
    prefix_head: usize,
}

// Prefix cuts deliberately retain their backing vectors on the hot path so a
// queue can be drained in constant time.  GC is a better place to pay the
// compaction cost, but only after enough dead prefix has accumulated to repay
// the copy and allocator churn.
const GC_LIST_PREFIX_MIN_ENTRIES: usize = 1_024;
const GC_LIST_PREFIX_MIN_BYTES: usize = 256 * 1_024;
const GC_LIST_PREFIX_RATIO_DENOMINATOR: usize = 4;
// Boot203 reached Lighting with 10.23m spare payload slots and 6.85m spare
// order slots spread across the live list population. Requiring one vector to
// waste 256 KiB reclaimed only a single vector, despite roughly 260 MiB of
// aggregate slack. Reclaim medium/large vectors once they are at least 50%
// over live length. Small vectors shrink exactly because their aggregate
// waste dominates; page-scale vectors retain 25% growth headroom so a
// still-growing startup list does not immediately reallocate after the GC.

/// Aggregate backing-storage telemetry collected while live lists are swept.
///
/// Lengths and capacities describe physical storage after GC compaction.  A
/// retained prefix is logically absent but still occupies `payload` and
/// `order` storage until it becomes large enough to compact.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ListStorageStats {
    /// Live list identities with allocated backing storage.
    pub allocated_lists: usize,
    /// Physical [`Value`] slots in positional and associative storage.
    pub payload_len: usize,
    /// Allocated [`Value`] capacity in positional and associative storage.
    pub payload_capacity: usize,
    /// Physical unified-order entries.
    pub order_len: usize,
    /// Allocated unified-order capacity.
    pub order_capacity: usize,
    /// Logically removed prefix entries deliberately retained after this GC.
    pub prefix_retained: usize,
    /// Lists whose lazy prefix was materialized by this GC.
    pub compacted_lists: usize,
    /// Logically removed entries physically released by this GC.
    pub compacted_prefix_entries: usize,
    /// Backing vectors whose gross excess capacity was released.
    pub shrunk_vectors: usize,
    /// Capacity bytes returned by successful vector shrinks.
    pub reclaimed_capacity_bytes: usize,
    /// Eligible vectors retained because their backing storage is shared COW.
    pub shared_shrink_candidates: usize,
    /// Retained associative lookup indexes.
    pub associative_indexes: usize,
    /// Entries across retained associative lookup indexes.
    pub associative_index_len: usize,
    /// Hash-table capacity across retained associative lookup indexes.
    pub associative_index_capacity: usize,
    /// Index counts in capacity/length bins: `<1.125`, `<1.25`, `<1.5`,
    /// `<2.0`, and `>=2.0`.
    pub associative_index_ratio_bins: [usize; 5],
    /// Associative indexes whose excess bucket capacity was released.
    pub shrunk_associative_indexes: usize,
    /// Hash-table capacity slots released from associative indexes.
    pub reclaimed_associative_index_capacity: usize,
    /// Entry-storage bytes represented by released association-index capacity.
    ///
    /// This excludes the standard library hash table's private control-byte
    /// allocation, so it is a conservative measure of allocator bytes freed.
    pub reclaimed_associative_index_bytes: usize,
    /// Transient positional-remove indexes observed before dropping unique ones.
    pub positional_remove_indexes: usize,
    /// Semantic-key entries across positional-remove indexes.
    pub positional_remove_key_len: usize,
    /// Hash-table capacity across positional-remove indexes.
    pub positional_remove_key_capacity: usize,
    /// Stored source positions across positional-remove indexes.
    pub positional_remove_position_len: usize,
    /// Position-vector capacity across positional-remove indexes.
    pub positional_remove_position_capacity: usize,
    /// Fenwick bookkeeping length across positional-remove indexes.
    pub positional_remove_removed_len: usize,
    /// Fenwick bookkeeping capacity across positional-remove indexes.
    pub positional_remove_removed_capacity: usize,
    /// Unique transient positional-remove indexes released by this GC.
    pub dropped_positional_remove_indexes: usize,
    /// Derived-index reclamation candidates skipped to preserve shared COW.
    pub shared_derived_index_candidates: usize,
}

#[derive(Clone, Debug)]
struct PositionalRemoveIndex {
    positions: HashMap<SemanticKey, Vec<usize>>,
    removed: Vec<usize>,
}

impl PositionalRemoveIndex {
    fn removed_before(&self, original: usize) -> usize {
        let mut cursor = original;
        let mut total = 0;
        while cursor > 0 {
            total += self.removed[cursor];
            cursor &= cursor - 1;
        }
        total
    }

    fn mark_removed(&mut self, original: usize) {
        let mut cursor = original + 1;
        while cursor < self.removed.len() {
            self.removed[cursor] += 1;
            cursor += cursor & cursor.wrapping_neg();
        }
    }
}

impl std::ops::Deref for DmList {
    type Target = DmListStorage;

    fn deref(&self) -> &Self::Target {
        static EMPTY: std::sync::OnceLock<DmListStorage> = std::sync::OnceLock::new();
        self.storage
            .as_deref()
            .unwrap_or_else(|| EMPTY.get_or_init(DmListStorage::default))
    }
}

impl std::ops::DerefMut for DmList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(
            self.storage
                .get_or_insert_with(|| Arc::new(DmListStorage::default())),
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct ListOrder(u32);

impl ListOrder {
    const ASSOCIATIVE_BIT: u32 = 1 << 31;

    fn positional(index: usize) -> Self {
        Self(u32::try_from(index).expect("DM list positional storage exceeds 2^31 entries"))
    }

    fn associative(index: usize) -> Self {
        Self(
            u32::try_from(index).expect("DM list associative storage exceeds 2^31 entries")
                | Self::ASSOCIATIVE_BIT,
        )
    }

    const fn positional_index(self) -> Option<usize> {
        if self.0 & Self::ASSOCIATIVE_BIT == 0 {
            Some(self.0 as usize)
        } else {
            None
        }
    }

    const fn associative_index(self) -> Option<usize> {
        if self.0 & Self::ASSOCIATIVE_BIT != 0 {
            Some((self.0 & !Self::ASSOCIATIVE_BIT) as usize)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum SemanticKey {
    Null,
    Number(u32),
    Text(Arc<str>),
    File(Arc<str>),
    TypePath(TypePath),
    Datum(DatumId),
    List(ListId),
}

const ASSOCIATIVE_INDEX_COLLISION: u32 = u32::MAX;

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        self.0 = hash;
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type AssociativeIndex = HashMap<u64, u32, BuildHasherDefault<IdentityHasher>>;

fn semantic_key(value: &Value) -> Option<SemanticKey> {
    Some(match value {
        Value::Null => SemanticKey::Null,
        Value::Number(number) => {
            let value = number.to_f32();
            if value.is_nan() {
                return None;
            }
            SemanticKey::Number(if value == 0.0 { 0 } else { number.bits() })
        }
        Value::Text(text) => SemanticKey::Text(Arc::clone(text)),
        Value::File(path) => SemanticKey::File(Arc::clone(path)),
        Value::TypePath(path) => SemanticKey::TypePath(path.clone()),
        Value::Datum(datum) => SemanticKey::Datum(*datum),
        Value::List(list) => SemanticKey::List(*list),
        Value::ModifiedTypePath(_) => return None,
    })
}

fn semantic_hash(value: &Value) -> Option<u64> {
    let mut hasher = DefaultHasher::new();
    match value {
        Value::Null => 0_u8.hash(&mut hasher),
        Value::Number(number) => {
            let value = number.to_f32();
            if value.is_nan() {
                return None;
            }
            1_u8.hash(&mut hasher);
            (if value == 0.0 { 0 } else { number.bits() }).hash(&mut hasher);
        }
        Value::Text(text) => {
            2_u8.hash(&mut hasher);
            text.as_ref().hash(&mut hasher);
        }
        Value::File(path) => {
            3_u8.hash(&mut hasher);
            path.as_ref().hash(&mut hasher);
        }
        Value::TypePath(path) => {
            4_u8.hash(&mut hasher);
            path.as_str().hash(&mut hasher);
        }
        Value::Datum(datum) => {
            5_u8.hash(&mut hasher);
            datum.hash(&mut hasher);
        }
        Value::List(list) => {
            6_u8.hash(&mut hasher);
            list.hash(&mut hasher);
        }
        Value::ModifiedTypePath(_) => return None,
    }
    Some(hasher.finish())
}

impl DmList {
    const ASSOCIATIVE_INDEX_THRESHOLD: usize = 8;
    const POSITIONAL_REMOVE_INDEX_THRESHOLD: usize = 64;

    /// Reserves capacity for positional values without changing list length.
    ///
    /// This is an engine allocation hint only; DM ordering and indexing remain
    /// unchanged.
    pub fn reserve_positional(&mut self, additional: usize) {
        self.positional.reserve(additional);
        self.order.reserve(additional);
    }

    fn should_compact_prefix_for_gc(&self) -> bool {
        let head = self.prefix_head;
        if head < GC_LIST_PREFIX_MIN_ENTRIES {
            return false;
        }
        let prefix_bytes = head.saturating_mul(
            std::mem::size_of::<Value>().saturating_add(std::mem::size_of::<ListOrder>()),
        );
        head.saturating_mul(GC_LIST_PREFIX_RATIO_DENOMINATOR) >= self.positional.len()
            || prefix_bytes >= GC_LIST_PREFIX_MIN_BYTES
    }

    fn compact_and_measure_for_gc(&mut self, aggregate: &mut ListStorageStats) {
        let compacted_prefix = self
            .should_compact_prefix_for_gc()
            .then_some(self.prefix_head);
        if let Some(prefix) = compacted_prefix {
            // A list identity can share storage with a shallow COW copy.  Do
            // not clone its dead prefix merely to drain it: construct this
            // identity directly from the logical suffix while leaving the
            // other identity untouched.
            if self
                .storage
                .as_ref()
                .is_some_and(|storage| Arc::strong_count(storage) > 1)
            {
                let source = self
                    .storage
                    .as_deref()
                    .expect("a non-zero prefix has allocated storage");
                debug_assert!(source.associative.is_empty());
                let positional = source.positional[prefix..].to_vec();
                let order = (0..positional.len()).map(ListOrder::positional).collect();
                self.storage = Some(Arc::new(DmListStorage {
                    positional,
                    associative: Vec::new(),
                    order,
                    associative_index: None,
                    positional_remove_index: None,
                    prefix_head: 0,
                }));
            } else {
                self.materialize_prefix();
            }
            aggregate.compacted_lists = aggregate.compacted_lists.saturating_add(1);
            aggregate.compacted_prefix_entries =
                aggregate.compacted_prefix_entries.saturating_add(prefix);
        }

        // Capacity slack is independent of lazy-prefix compaction. Most
        // Boot203 candidates had no retained prefix at all, so inspect every
        // live allocated list during the sweep we already perform. Never call
        // Arc::make_mut here: cloning a shared COW payload merely to trim its
        // capacity would multiply live memory and break the sharing win.
        if let Some(storage) = self.storage.as_mut() {
            let shrink_candidates = [
                gc_vector_shrink_target(&storage.positional).is_some(),
                gc_vector_shrink_target(&storage.associative).is_some(),
                gc_vector_shrink_target(&storage.order).is_some(),
            ]
            .into_iter()
            .filter(|candidate| *candidate)
            .count();
            if shrink_candidates > 0 && Arc::strong_count(storage) > 1 {
                aggregate.shared_shrink_candidates = aggregate
                    .shared_shrink_candidates
                    .saturating_add(shrink_candidates);
            } else if shrink_candidates > 0 {
                let storage = Arc::get_mut(storage)
                    .expect("a uniquely owned list storage must be mutably accessible");
                for reclaimed in [
                    gc_shrink_vector(&mut storage.positional),
                    gc_shrink_vector(&mut storage.associative),
                    gc_shrink_vector(&mut storage.order),
                ] {
                    aggregate.shrunk_vectors = aggregate
                        .shrunk_vectors
                        .saturating_add(usize::from(reclaimed > 0));
                    aggregate.reclaimed_capacity_bytes =
                        aggregate.reclaimed_capacity_bytes.saturating_add(reclaimed);
                }
            }

            let positional_index_present = storage.positional_remove_index.is_some();
            if let Some(index) = storage.positional_remove_index.as_deref() {
                aggregate.positional_remove_indexes =
                    aggregate.positional_remove_indexes.saturating_add(1);
                aggregate.positional_remove_key_len = aggregate
                    .positional_remove_key_len
                    .saturating_add(index.positions.len());
                aggregate.positional_remove_key_capacity = aggregate
                    .positional_remove_key_capacity
                    .saturating_add(index.positions.capacity());
                for positions in index.positions.values() {
                    aggregate.positional_remove_position_len = aggregate
                        .positional_remove_position_len
                        .saturating_add(positions.len());
                    aggregate.positional_remove_position_capacity = aggregate
                        .positional_remove_position_capacity
                        .saturating_add(positions.capacity());
                }
                aggregate.positional_remove_removed_len = aggregate
                    .positional_remove_removed_len
                    .saturating_add(index.removed.len());
                aggregate.positional_remove_removed_capacity = aggregate
                    .positional_remove_removed_capacity
                    .saturating_add(index.removed.capacity());
            }

            let associative_shrink_candidate =
                storage.associative_index.as_deref().is_some_and(|index| {
                    gc_index_shrink_target(index.len(), index.capacity()).is_some()
                });
            if Arc::strong_count(storage) > 1 {
                aggregate.shared_derived_index_candidates = aggregate
                    .shared_derived_index_candidates
                    .saturating_add(usize::from(positional_index_present))
                    .saturating_add(usize::from(associative_shrink_candidate));
            } else {
                let storage = Arc::get_mut(storage)
                    .expect("a uniquely owned list storage must be mutably accessible");
                if let Some(index) = &mut storage.associative_index
                    && let Some(target) = gc_index_shrink_target(index.len(), index.capacity())
                {
                    let before = index.capacity();
                    index.shrink_to(target);
                    let reclaimed = before.saturating_sub(index.capacity());
                    aggregate.shrunk_associative_indexes = aggregate
                        .shrunk_associative_indexes
                        .saturating_add(usize::from(reclaimed > 0));
                    aggregate.reclaimed_associative_index_capacity = aggregate
                        .reclaimed_associative_index_capacity
                        .saturating_add(reclaimed);
                    aggregate.reclaimed_associative_index_bytes =
                        aggregate.reclaimed_associative_index_bytes.saturating_add(
                            reclaimed.saturating_mul(std::mem::size_of::<(u64, u32)>()),
                        );
                }
                if storage.positional_remove_index.take().is_some() {
                    aggregate.dropped_positional_remove_indexes = aggregate
                        .dropped_positional_remove_indexes
                        .saturating_add(1);
                }
            }
        }

        let Some(storage) = self.storage.as_deref() else {
            return;
        };
        aggregate.allocated_lists = aggregate.allocated_lists.saturating_add(1);
        aggregate.payload_len = aggregate
            .payload_len
            .saturating_add(storage.positional.len())
            .saturating_add(storage.associative.len().saturating_mul(2));
        aggregate.payload_capacity = aggregate
            .payload_capacity
            .saturating_add(storage.positional.capacity())
            .saturating_add(storage.associative.capacity().saturating_mul(2));
        aggregate.order_len = aggregate.order_len.saturating_add(storage.order.len());
        aggregate.order_capacity = aggregate
            .order_capacity
            .saturating_add(storage.order.capacity());
        aggregate.prefix_retained = aggregate
            .prefix_retained
            .saturating_add(storage.prefix_head);
        if let Some(index) = storage.associative_index.as_deref() {
            aggregate.associative_indexes = aggregate.associative_indexes.saturating_add(1);
            let ratio_bin = gc_index_ratio_bin(index.len(), index.capacity());
            aggregate.associative_index_ratio_bins[ratio_bin] =
                aggregate.associative_index_ratio_bins[ratio_bin].saturating_add(1);
            aggregate.associative_index_len =
                aggregate.associative_index_len.saturating_add(index.len());
            aggregate.associative_index_capacity = aggregate
                .associative_index_capacity
                .saturating_add(index.capacity());
        }
    }

    fn materialize_prefix(&mut self) {
        let head = self.prefix_head;
        if head == 0 {
            return;
        }
        let storage = &mut **self;
        storage.positional.drain(..head);
        storage.order.drain(..head);
        for entry in &mut storage.order {
            let index = entry
                .positional_index()
                .expect("lazy list prefix is only used for positional lists");
            *entry = ListOrder::positional(index - head);
        }
        storage.prefix_head = 0;
    }

    fn invalidate_positional_remove_index(&mut self) {
        if let Some(storage) = &mut self.storage {
            Arc::make_mut(storage).positional_remove_index = None;
        }
    }

    fn rebuild_associative_index(&mut self) {
        if self.associative.len() < Self::ASSOCIATIVE_INDEX_THRESHOLD {
            self.associative_index = None;
            return;
        }
        let mut index = AssociativeIndex::with_capacity_and_hasher(
            self.associative.len(),
            BuildHasherDefault::default(),
        );
        for (position, (key, _)) in self.associative.iter().enumerate() {
            let Some(hash) = semantic_hash(key) else {
                continue;
            };
            let compact_position =
                u32::try_from(position).expect("DM list associative storage exceeds 2^31 entries");
            match index.entry(hash) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(compact_position);
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let indexed = *entry.get();
                    if indexed != ASSOCIATIVE_INDEX_COLLISION
                        && !self.associative[indexed as usize].0.semantic_eq(key)
                    {
                        entry.insert(ASSOCIATIVE_INDEX_COLLISION);
                    }
                }
            }
        }
        self.associative_index = Some(Box::new(index));
    }

    fn index_last_association(&mut self) {
        if self.associative.len() == Self::ASSOCIATIVE_INDEX_THRESHOLD {
            self.rebuild_associative_index();
            return;
        }
        if self.associative.len() < Self::ASSOCIATIVE_INDEX_THRESHOLD {
            return;
        }
        let position = self.associative.len() - 1;
        let Some(hash) = semantic_hash(&self.associative[position].0) else {
            return;
        };
        let compact_position =
            u32::try_from(position).expect("DM list associative storage exceeds 2^31 entries");
        let collision = self
            .associative_index
            .as_ref()
            .and_then(|index| index.get(&hash))
            .copied()
            .is_some_and(|indexed| {
                indexed != ASSOCIATIVE_INDEX_COLLISION
                    && !self.associative[indexed as usize]
                        .0
                        .semantic_eq(&self.associative[position].0)
            });
        if let Some(index) = &mut self.associative_index {
            match index.entry(hash) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(compact_position);
                }
                std::collections::hash_map::Entry::Occupied(mut entry) if collision => {
                    entry.insert(ASSOCIATIVE_INDEX_COLLISION);
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }
    }

    fn association_position_for_key(&self, key: &Value) -> Option<usize> {
        match (&self.associative_index, semantic_hash(key)) {
            (Some(index), Some(hash)) => match index.get(&hash).copied() {
                Some(position) if position != ASSOCIATIVE_INDEX_COLLISION => {
                    let position = position as usize;
                    if self
                        .associative
                        .get(position)
                        .is_some_and(|(candidate, _)| candidate.semantic_eq(key))
                    {
                        Some(position)
                    } else {
                        self.associative
                            .iter()
                            .position(|(candidate, _)| candidate.semantic_eq(key))
                    }
                }
                Some(_) => self
                    .associative
                    .iter()
                    .position(|(candidate, _)| candidate.semantic_eq(key)),
                None => None,
            },
            _ => self
                .associative
                .iter()
                .position(|(candidate, _)| candidate.semantic_eq(key)),
        }
    }

    /// Returns the number of values and associative keys in iteration order.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len() - self.prefix_head
    }

    /// Returns the number of entries without an associative value.
    #[must_use]
    pub fn positional_len(&self) -> usize {
        self.positional.len() - self.prefix_head
    }

    /// Returns the number of associative entries.
    #[must_use]
    pub fn associative_len(&self) -> usize {
        self.associative.len()
    }

    /// Returns whether there are no positional or associative entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends a positional value and returns its 1-based index.
    pub fn add(&mut self, value: Value) -> usize {
        self.invalidate_positional_remove_index();
        let position = self.positional.len();
        self.order.push(ListOrder::positional(position));
        self.positional.push(value);
        self.len()
    }

    /// Appends positional values in iterator order with one storage mutation.
    ///
    /// This is semantically identical to repeated [`Self::add`] calls, but it
    /// invalidates derived indexes and detaches shared copy-on-write storage
    /// only once. Large engine-produced lists such as `block()` rectangles can
    /// therefore fill both payload and iteration-order vectors in bulk.
    /// Returns the resulting DM-visible list length.
    pub fn extend_positional(&mut self, values: impl IntoIterator<Item = Value>) -> usize {
        self.invalidate_positional_remove_index();
        let mut values = values.into_iter();
        // The lower bound cannot over-reserve even for a custom iterator that
        // reports an untrusted upper hint. Exact-size Vec iterators still give
        // us the full one-shot reservation used by engine bulk fills.
        let (reserve, _) = values.size_hint();
        let storage = &mut **self;
        storage.positional.reserve(reserve);
        storage.order.reserve(reserve);
        let first = storage.positional.len();
        storage.positional.extend(&mut values);
        let end = storage.positional.len();
        storage
            .order
            .extend((first..end).map(ListOrder::positional));
        storage.order.len() - storage.prefix_head
    }

    /// Reads a 1-based positional entry.
    ///
    /// # Errors
    ///
    /// Returns a precise index error for zero or an index beyond `len`.
    pub fn get(&self, index: usize) -> Result<&Value, ValueError> {
        let zero_based = checked_index(index, self.len())? + self.prefix_head;
        let entry = self.order[zero_based];
        Ok(if let Some(position) = entry.positional_index() {
            &self.positional[position]
        } else {
            &self.associative[entry
                .associative_index()
                .expect("validated associative order")]
            .0
        })
    }

    /// Replaces a 1-based positional entry and returns its previous value.
    ///
    /// # Errors
    ///
    /// Returns a precise index error for zero or an index beyond `len`.
    pub fn set(&mut self, index: usize, value: Value) -> Result<Value, ValueError> {
        let zero_based = checked_index(index, self.len())? + self.prefix_head;
        let Some(position) = self.order[zero_based].positional_index() else {
            return Err(ValueError::AssociativeIndexAssignment { index });
        };
        self.invalidate_positional_remove_index();
        Ok(std::mem::replace(&mut self.positional[position], value))
    }

    /// Removes and returns a 1-based positional entry.
    ///
    /// # Errors
    ///
    /// Returns a precise index error for zero or an index beyond `len`.
    pub fn remove(&mut self, index: usize) -> Result<Value, ValueError> {
        self.materialize_prefix();
        self.invalidate_positional_remove_index();
        let zero_based = checked_index(index, self.order.len())?;
        let entry = self.order.remove(zero_based);
        if let Some(position) = entry.positional_index() {
            for entry in &mut self.order {
                if let Some(other) = entry.positional_index()
                    && other > position
                {
                    *entry = ListOrder::positional(other - 1);
                }
            }
            Ok(self.positional.remove(position))
        } else {
            let association = entry
                .associative_index()
                .ok_or(ValueError::CorruptListStorage)?;
            let (key, _) = self.associative.remove(association);
            for entry in &mut self.order {
                if let Some(other) = entry.associative_index()
                    && other > association
                {
                    *entry = ListOrder::associative(other - 1);
                }
            }
            self.rebuild_associative_index();
            Ok(key)
        }
    }

    /// Removes the first position equal to `value` under [`Value::semantic_eq`].
    pub fn remove_first(&mut self, value: &Value) -> Option<Value> {
        let index = self
            .positions()
            .position(|(_, candidate)| candidate.semantic_eq(value))?;
        self.remove(index + 1).ok()
    }

    /// Inserts a positional value at a 1-based boundary and returns that index.
    ///
    /// # Errors
    ///
    /// Returns an index error when `index` is zero or greater than `len + 1`.
    pub fn insert(&mut self, index: usize, value: Value) -> Result<usize, ValueError> {
        self.materialize_prefix();
        checked_boundary(index, self.order.len())?;
        self.invalidate_positional_remove_index();
        let position = self.positional.len();
        self.positional.push(value);
        self.order
            .insert(index - 1, ListOrder::positional(position));
        Ok(index)
    }

    /// Creates a shallow copy of the half-open 1-based range `[start, end)`.
    ///
    /// Associative values remain associated with copied keys.
    ///
    /// # Errors
    ///
    /// Returns an index error for boundaries outside `1..=len + 1`, or
    /// [`ValueError::CorruptListStorage`] if an associative order entry lost
    /// its value.
    pub fn copy_range(&self, start: usize, end: usize) -> Result<Self, ValueError> {
        checked_boundary(start, self.len())?;
        checked_boundary(end, self.len())?;
        let mut copy = Self::default();
        if end <= start {
            return Ok(copy);
        }
        let offset = self.prefix_head;
        for entry in &self.order[offset + start - 1..offset + end - 1] {
            if let Some(position) = entry.positional_index() {
                copy.add(self.positional[position].clone());
            } else {
                let association = entry
                    .associative_index()
                    .ok_or(ValueError::CorruptListStorage)?;
                let (key, value) = self
                    .associative
                    .get(association)
                    .ok_or(ValueError::CorruptListStorage)?;
                copy.set_key(key.clone(), value.clone());
            }
        }
        Ok(copy)
    }

    /// Removes the half-open 1-based range `[start, end)` and returns the
    /// number of removed iteration entries.
    ///
    /// # Errors
    ///
    /// Returns an index error for boundaries outside `1..=len + 1`.
    pub fn cut_range(&mut self, start: usize, end: usize) -> Result<usize, ValueError> {
        let logical_len = self.len();
        checked_boundary(start, logical_len)?;
        checked_boundary(end, logical_len)?;
        if end <= start {
            return Ok(0);
        }
        let count = end - start;

        // `Cut()` is a very common list-clearing idiom in DM. Release the
        // lazily allocated backing store in one operation instead of repeatedly
        // shifting its three parallel vectors.
        if start == 1 && end == logical_len + 1 {
            self.storage = None;
            return Ok(count);
        }

        if start == 1
            && self.associative.is_empty()
            && (self.prefix_head > 0
                || self
                    .order
                    .iter()
                    .enumerate()
                    .all(|(index, entry)| entry.positional_index() == Some(index)))
        {
            self.invalidate_positional_remove_index();
            self.prefix_head += count;
            return Ok(count);
        }

        self.materialize_prefix();

        self.invalidate_positional_remove_index();

        let storage = Arc::make_mut(
            self.storage
                .as_mut()
                .ok_or(ValueError::CorruptListStorage)?,
        );
        let removed = storage.order.drain(start - 1..end - 1);
        let mut remove_positional = vec![false; storage.positional.len()];
        let mut remove_associative = vec![false; storage.associative.len()];
        for entry in removed {
            if let Some(index) = entry.positional_index() {
                *remove_positional
                    .get_mut(index)
                    .ok_or(ValueError::CorruptListStorage)? = true;
            } else {
                let index = entry
                    .associative_index()
                    .ok_or(ValueError::CorruptListStorage)?;
                *remove_associative
                    .get_mut(index)
                    .ok_or(ValueError::CorruptListStorage)? = true;
            }
        }

        let mut positional_remap = vec![usize::MAX; storage.positional.len()];
        let mut next = 0;
        for (old, removed) in remove_positional.iter().copied().enumerate() {
            if !removed {
                positional_remap[old] = next;
                next += 1;
            }
        }
        storage.positional = std::mem::take(&mut storage.positional)
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| (!remove_positional[index]).then_some(value))
            .collect();

        let mut associative_remap = vec![usize::MAX; storage.associative.len()];
        next = 0;
        for (old, removed) in remove_associative.iter().copied().enumerate() {
            if !removed {
                associative_remap[old] = next;
                next += 1;
            }
        }
        storage.associative = std::mem::take(&mut storage.associative)
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| (!remove_associative[index]).then_some(value))
            .collect();

        for entry in &mut storage.order {
            *entry = if let Some(index) = entry.positional_index() {
                ListOrder::positional(positional_remap[index])
            } else {
                ListOrder::associative(
                    associative_remap[entry
                        .associative_index()
                        .ok_or(ValueError::CorruptListStorage)?],
                )
            };
        }
        self.rebuild_associative_index();
        Ok(count)
    }

    /// Finds the first iteration position semantically equal to `value` in
    /// the half-open 1-based range `[start, end)`, returning zero when absent.
    ///
    /// # Errors
    ///
    /// Returns an index error for boundaries outside `1..=len + 1`.
    pub fn find_position(
        &self,
        value: &Value,
        start: usize,
        end: usize,
    ) -> Result<usize, ValueError> {
        checked_boundary(start, self.len())?;
        checked_boundary(end, self.len())?;
        if end <= start {
            return Ok(0);
        }
        for index in start..end {
            if self.get(index)?.semantic_eq(value) {
                return Ok(index);
            }
        }
        Ok(0)
    }

    /// Removes the last occurrence equal to `value`, matching BYOND list
    /// subtraction/`Remove()` ordering.
    pub fn remove_last(&mut self, value: &Value) -> Option<Value> {
        self.materialize_prefix();
        if self.positional_remove_index.is_none()
            && self.associative.is_empty()
            && self.positional.len() >= Self::POSITIONAL_REMOVE_INDEX_THRESHOLD
            && self.order.iter().enumerate().all(|(index, entry)| {
                entry.positional_index() == Some(index)
                    && semantic_key(&self.positional[index]).is_some()
            })
        {
            let mut positions: HashMap<SemanticKey, Vec<usize>> = HashMap::new();
            for (index, value) in self.positional.iter().enumerate() {
                positions
                    .entry(semantic_key(value)?)
                    .or_default()
                    .push(index);
            }
            self.positional_remove_index = Some(Box::new(PositionalRemoveIndex {
                positions,
                removed: vec![0; self.positional.len() + 1],
            }));
        }

        if let (Some(key), Some(index)) =
            (semantic_key(value), self.positional_remove_index.as_mut())
        {
            let original = index.positions.get_mut(&key)?.pop()?;
            let current = original - index.removed_before(original);
            index.mark_removed(original);
            self.order.pop();
            return Some(self.positional.remove(current));
        }

        let index = (1..=self.len()).rev().find(|index| {
            self.get(*index)
                .is_ok_and(|candidate| candidate.semantic_eq(value))
        })?;
        self.remove(index).ok()
    }

    /// Removes the last matching occurrence for every entry in `values`.
    ///
    /// This is the bulk form of repeated [`Self::remove_last`]. It preserves
    /// BYOND's multiset subtraction semantics while compacting positional,
    /// associative, and unified-order storage once instead of shifting them
    /// once per right-hand entry.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::CorruptListStorage`] if an order entry no longer
    /// refers to a live positional or associative value.
    pub fn subtract_entries(&mut self, values: &[Value]) -> Result<usize, ValueError> {
        if values.is_empty() || self.len() == 0 {
            return Ok(0);
        }

        let mut keyed_quotas = HashMap::<SemanticKey, usize>::new();
        let mut fallback_quotas = Vec::<(Value, usize)>::new();
        for value in values {
            if let Some(key) = semantic_key(value) {
                *keyed_quotas.entry(key).or_default() += 1;
            } else if let Some((_, count)) = fallback_quotas
                .iter_mut()
                .find(|(candidate, _)| candidate.semantic_eq(value))
            {
                *count += 1;
            } else {
                fallback_quotas.push((value.clone(), 1));
            }
        }

        let mut remove_order = vec![false; self.order.len()];
        let mut removed = 0;
        for order_index in (self.prefix_head..self.order.len()).rev() {
            let entry = self.order[order_index];
            let candidate = if let Some(index) = entry.positional_index() {
                self.positional
                    .get(index)
                    .ok_or(ValueError::CorruptListStorage)?
            } else {
                let index = entry
                    .associative_index()
                    .ok_or(ValueError::CorruptListStorage)?;
                &self
                    .associative
                    .get(index)
                    .ok_or(ValueError::CorruptListStorage)?
                    .0
            };
            let matches = if let Some(key) = semantic_key(candidate) {
                keyed_quotas.get_mut(&key).is_some_and(|count| {
                    if *count == 0 {
                        false
                    } else {
                        *count -= 1;
                        true
                    }
                })
            } else {
                fallback_quotas
                    .iter_mut()
                    .find(|(value, count)| *count > 0 && candidate.semantic_eq(value))
                    .is_some_and(|(_, count)| {
                        *count -= 1;
                        true
                    })
            };
            if matches {
                remove_order[order_index] = true;
                removed += 1;
            }
        }
        if removed == 0 {
            return Ok(0);
        }

        self.invalidate_positional_remove_index();
        let storage = Arc::make_mut(
            self.storage
                .as_mut()
                .ok_or(ValueError::CorruptListStorage)?,
        );
        let mut remove_positional = vec![false; storage.positional.len()];
        let mut remove_associative = vec![false; storage.associative.len()];
        for (order_index, entry) in storage.order.iter().copied().enumerate() {
            if order_index >= storage.prefix_head && !remove_order[order_index] {
                continue;
            }
            if let Some(index) = entry.positional_index() {
                *remove_positional
                    .get_mut(index)
                    .ok_or(ValueError::CorruptListStorage)? = true;
            } else {
                let index = entry
                    .associative_index()
                    .ok_or(ValueError::CorruptListStorage)?;
                *remove_associative
                    .get_mut(index)
                    .ok_or(ValueError::CorruptListStorage)? = true;
            }
        }

        let mut positional_remap = vec![usize::MAX; storage.positional.len()];
        let mut next = 0;
        for (old, remove) in remove_positional.iter().copied().enumerate() {
            if !remove {
                positional_remap[old] = next;
                next += 1;
            }
        }
        storage.positional = std::mem::take(&mut storage.positional)
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| (!remove_positional[index]).then_some(value))
            .collect();

        let mut associative_remap = vec![usize::MAX; storage.associative.len()];
        next = 0;
        for (old, remove) in remove_associative.iter().copied().enumerate() {
            if !remove {
                associative_remap[old] = next;
                next += 1;
            }
        }
        storage.associative = std::mem::take(&mut storage.associative)
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| (!remove_associative[index]).then_some(value))
            .collect();

        storage.order = std::mem::take(&mut storage.order)
            .into_iter()
            .enumerate()
            .filter_map(|(order_index, entry)| {
                (order_index >= storage.prefix_head && !remove_order[order_index]).then(|| {
                    if let Some(index) = entry.positional_index() {
                        ListOrder::positional(positional_remap[index])
                    } else {
                        ListOrder::associative(
                            associative_remap[entry
                                .associative_index()
                                .expect("validated associative order entry")],
                        )
                    }
                })
            })
            .collect();
        storage.prefix_head = 0;
        storage.positional_remove_index = None;
        self.rebuild_associative_index();
        Ok(removed)
    }

    /// Swaps two 1-based iteration positions while keeping associative values
    /// attached to their keys.
    ///
    /// # Errors
    ///
    /// Returns an index error when either position is outside the list.
    pub fn swap(&mut self, first: usize, second: usize) -> Result<(), ValueError> {
        self.materialize_prefix();
        let first = checked_index(first, self.order.len())?;
        let second = checked_index(second, self.order.len())?;
        self.invalidate_positional_remove_index();
        self.order.swap(first, second);
        Ok(())
    }

    /// Resizes the list, appending positional `null` values when growing and
    /// cutting the tail when shrinking.
    ///
    /// # Errors
    ///
    /// Returns a storage error only if an existing associative entry is
    /// internally inconsistent while shrinking.
    pub fn resize(&mut self, new_len: usize) -> Result<(), ValueError> {
        while self.len() < new_len {
            self.add(Value::Null);
        }
        if self.len() > new_len {
            let end = self.len() + 1;
            self.cut_range(new_len + 1, end)?;
        }
        Ok(())
    }

    /// Reads an associative value by semantic key equality.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::MissingKey`] when the key is absent.
    pub fn get_key(&self, key: &Value) -> Result<&Value, ValueError> {
        self.association_position_for_key(key)
            .and_then(|position| self.associative.get(position))
            .map(|(_, value)| value)
            .ok_or(ValueError::MissingKey)
    }

    /// Inserts or replaces an associative value.
    ///
    /// Replacing a key retains its deterministic insertion position and
    /// returns the old value. New keys return `None`.
    pub fn set_key(&mut self, key: Value, value: Value) -> Option<Value> {
        self.materialize_prefix();
        self.invalidate_positional_remove_index();
        let existing = self.association_position_for_key(&key);
        if let Some(position) = existing {
            let current = &mut self.associative[position].1;
            return Some(std::mem::replace(current, value));
        }
        if let Some((order_index, position)) =
            self.order.iter().enumerate().find_map(|(index, entry)| {
                let Some(position) = entry.positional_index() else {
                    return None;
                };
                self.positional[position]
                    .semantic_eq(&key)
                    .then_some((index, position))
            })
        {
            let existing_key = self.positional.remove(position);
            for entry in &mut self.order {
                if let Some(other) = entry.positional_index()
                    && other > position
                {
                    *entry = ListOrder::positional(other - 1);
                }
            }
            self.order[order_index] = ListOrder::associative(self.associative.len());
            self.associative.push((existing_key, value));
            self.index_last_association();
            return None;
        }
        let position = self.associative.len();
        self.order.push(ListOrder::associative(position));
        self.associative.push((key, value));
        self.index_last_association();
        None
    }

    /// Returns whether an iteration entry is semantically equal to `value`.
    #[must_use]
    pub fn contains(&self, value: &Value) -> bool {
        self.association_position_for_key(value).is_some()
            || self.positional[self.prefix_head..]
                .iter()
                .any(|candidate| candidate.semantic_eq(value))
    }

    /// Removes an associative key and returns its value.
    pub fn remove_key(&mut self, key: &Value) -> Option<Value> {
        self.materialize_prefix();
        self.invalidate_positional_remove_index();
        let index = self
            .associative
            .iter()
            .position(|(candidate, _)| candidate.semantic_eq(key))?;
        let order_index = self
            .order
            .iter()
            .position(|entry| entry.associative_index() == Some(index))?;
        let (_, value) = self.associative.remove(index);
        self.order.remove(order_index);
        for entry in &mut self.order {
            if let Some(other) = entry.associative_index()
                && other > index
            {
                *entry = ListOrder::associative(other - 1);
            }
        }
        self.rebuild_associative_index();
        Some(value)
    }

    /// Iterates positional entries in ascending 1-based index order.
    #[must_use]
    pub fn positions(&self) -> impl ExactSizeIterator<Item = (usize, &Value)> {
        self.order[self.prefix_head..]
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let value = if let Some(position) = entry.positional_index() {
                    &self.positional[position]
                } else {
                    &self.associative[entry.associative_index().expect("valid list order")].0
                };
                (index + 1, value)
            })
    }

    /// Iterates associative entries in stable key-insertion order.
    #[must_use]
    pub fn associations(&self) -> impl ExactSizeIterator<Item = (&Value, &Value)> {
        self.associative.iter().map(|(key, value)| (key, value))
    }
}

/// Heap and list operation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValueError {
    /// A ready-world heap image violated an arena or record invariant.
    CorruptHeapSnapshot(String),
    /// A type path was not a valid canonical absolute path.
    InvalidTypePath(String),
    /// A field name was not a canonical DM identifier.
    InvalidFieldName(String),
    /// A requested datum field is not materialized.
    MissingField(FieldName),
    /// Positional index zero was used where DM indices begin at one.
    IndexZero,
    /// Positional index was beyond the current list length.
    IndexOutOfBounds {
        /// Attempted 1-based index.
        index: usize,
        /// Current positional length.
        len: usize,
    },
    /// Associative key was absent.
    MissingKey,
    /// A runtime value could not represent a valid 1-based list index.
    InvalidListIndex(String),
    /// Internal ordered and associative storage lost synchronization.
    CorruptListStorage,
    /// Numeric assignment targeted an associative key's iteration position.
    AssociativeIndexAssignment {
        /// Attempted 1-based index.
        index: usize,
    },
    /// List identity is stale or does not belong to a live slot.
    StaleList(ListId),
    /// Datum identity is stale or does not belong to a live slot.
    StaleDatum(DatumId),
}

impl fmt::Display for ValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CorruptHeapSnapshot(message) => {
                write!(formatter, "corrupt ready-world heap snapshot: {message}")
            }
            Self::InvalidTypePath(path) => write!(formatter, "invalid absolute type path {path:?}"),
            Self::InvalidFieldName(name) => write!(formatter, "invalid DM field name {name:?}"),
            Self::MissingField(name) => write!(formatter, "datum field {name:?} is absent"),
            Self::IndexZero => {
                formatter.write_str("DM list position 0 is invalid; indices begin at 1")
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(formatter, "DM list position {index} exceeds length {len}")
            }
            Self::MissingKey => formatter.write_str("associative list key is absent"),
            Self::InvalidListIndex(message) => formatter.write_str(message),
            Self::CorruptListStorage => formatter.write_str("DM list storage is inconsistent"),
            Self::AssociativeIndexAssignment { index } => write!(
                formatter,
                "DM list position {index} is an associative key; assign through the key"
            ),
            Self::StaleList(id) => write!(formatter, "stale list handle {id:?}"),
            Self::StaleDatum(id) => write!(formatter, "stale datum handle {id:?}"),
        }
    }
}

impl std::error::Error for ValueError {}

fn checked_boundary(index: usize, len: usize) -> Result<usize, ValueError> {
    if index == 0 {
        return Err(ValueError::IndexZero);
    }
    if index > len.saturating_add(1) {
        return Err(ValueError::IndexOutOfBounds { index, len });
    }
    Ok(index - 1)
}

fn checked_index(index: usize, len: usize) -> Result<usize, ValueError> {
    if index == 0 {
        return Err(ValueError::IndexZero);
    }
    if index > len {
        return Err(ValueError::IndexOutOfBounds { index, len });
    }
    Ok(index - 1)
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

const ARENA_CHUNK_SLOTS: usize = 16 * 1024;

/// Constant-time slot-allocation telemetry for a value arena.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArenaStats {
    /// Slots currently containing a live value.
    pub live: usize,
    /// Slots that have been initialized at least once.
    pub slots: usize,
    /// Reusable vacant slots.
    pub free: usize,
    /// Individually allocated fixed-size slot chunks.
    pub chunks: usize,
    /// Slot capacity reserved across all chunks.
    pub reserved: usize,
}

struct Arena<T> {
    chunks: Vec<Vec<Slot<T>>>,
    slot_len: usize,
    free: Vec<u32>,
    live: usize,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            slot_len: 0,
            free: Vec::new(),
            live: 0,
        }
    }
}

impl<T> Arena<T> {
    fn push_snapshot_slot(&mut self, generation: u32, value: Option<T>) {
        if self
            .chunks
            .last()
            .is_none_or(|chunk| chunk.len() == ARENA_CHUNK_SLOTS)
        {
            self.chunks.push(Vec::with_capacity(ARENA_CHUNK_SLOTS));
        }
        self.live += usize::from(value.is_some());
        self.chunks
            .last_mut()
            .expect("arena snapshot chunk was just ensured")
            .push(Slot { generation, value });
        self.slot_len += 1;
    }

    fn install_snapshot_free(&mut self, free: Vec<u32>, kind: &str) -> Result<(), ValueError> {
        let mut free_seen = vec![false; self.slot_len];
        for &index in &free {
            let index = index as usize;
            if index >= self.slot_len
                || free_seen[index]
                || self.slot(index).is_some_and(|slot| slot.value.is_some())
            {
                return Err(ValueError::CorruptHeapSnapshot(format!(
                    "{kind} free list contains invalid slot {index}"
                )));
            }
            free_seen[index] = true;
        }
        for (index, seen) in free_seen.into_iter().enumerate() {
            let slot = self
                .slot(index)
                .expect("snapshot validation index addresses an arena slot");
            if slot.value.is_none() && slot.generation != u32::MAX && !seen {
                return Err(ValueError::CorruptHeapSnapshot(format!(
                    "{kind} vacant slot {index} is absent from its free list"
                )));
            }
        }
        self.free = free;
        Ok(())
    }

    fn stats(&self) -> ArenaStats {
        ArenaStats {
            live: self.live,
            slots: self.slot_len,
            free: self.free.len(),
            chunks: self.chunks.len(),
            reserved: self.chunks.len().saturating_mul(ARENA_CHUNK_SLOTS),
        }
    }

    fn slot(&self, index: usize) -> Option<&Slot<T>> {
        let chunk = index / ARENA_CHUNK_SLOTS;
        let offset = index % ARENA_CHUNK_SLOTS;
        self.chunks.get(chunk)?.get(offset)
    }

    fn slot_mut(&mut self, index: usize) -> Option<&mut Slot<T>> {
        let chunk = index / ARENA_CHUNK_SLOTS;
        let offset = index % ARENA_CHUNK_SLOTS;
        self.chunks.get_mut(chunk)?.get_mut(offset)
    }

    fn insert(&mut self, value: T) -> (u32, u32) {
        self.live += 1;
        if let Some(index) = self.free.pop() {
            let slot = self
                .slot_mut(index as usize)
                .expect("free arena index must address an allocated slot");
            debug_assert!(slot.value.is_none());
            slot.value = Some(value);
            return (index, slot.generation);
        }

        let index = u32::try_from(self.slot_len).expect("heap cannot exceed u32::MAX slots");
        if self
            .chunks
            .last()
            .is_none_or(|chunk| chunk.len() == ARENA_CHUNK_SLOTS)
        {
            self.chunks.push(Vec::with_capacity(ARENA_CHUNK_SLOTS));
        }
        self.chunks
            .last_mut()
            .expect("arena chunk was just ensured")
            .push(Slot {
                generation: 0,
                value: Some(value),
            });
        self.slot_len += 1;
        (index, 0)
    }

    fn get(&self, index: u32, generation: u32) -> Option<&T> {
        let slot = self.slot(index as usize)?;
        (slot.generation == generation)
            .then_some(slot.value.as_ref())
            .flatten()
    }

    fn get_mut(&mut self, index: u32, generation: u32) -> Option<&mut T> {
        let slot = self.slot_mut(index as usize)?;
        (slot.generation == generation)
            .then_some(slot.value.as_mut())
            .flatten()
    }

    fn remove(&mut self, index: u32, generation: u32) -> Option<T> {
        let (value, reusable) = {
            let slot = self.slot_mut(index as usize)?;
            if slot.generation != generation {
                return None;
            }
            let value = slot.value.take()?;
            let reusable = if let Some(next_generation) = slot.generation.checked_add(1) {
                slot.generation = next_generation;
                true
            } else {
                false
            };
            (value, reusable)
        };
        self.live -= 1;
        if reusable {
            self.free.push(index);
        }
        Some(value)
    }

    fn iter(&self) -> impl Iterator<Item = (u32, u32, &T)> {
        self.chunks
            .iter()
            .enumerate()
            .flat_map(|(chunk_index, chunk)| {
                chunk.iter().enumerate().filter_map(move |(offset, slot)| {
                    let index = chunk_index * ARENA_CHUNK_SLOTS + offset;
                    let index = u32::try_from(index).ok()?;
                    Some((index, slot.generation, slot.value.as_ref()?))
                })
            })
    }

    fn live_generation(&self, index: u32) -> Option<u32> {
        let slot = self.slot(index as usize)?;
        slot.value.as_ref().map(|_| slot.generation)
    }

    const fn len(&self) -> usize {
        self.live
    }

    const fn slot_len(&self) -> usize {
        self.slot_len
    }

    fn sweep_unmarked_with(
        &mut self,
        marked: &[bool],
        mut visit_retained: impl FnMut(&mut T),
    ) -> usize {
        debug_assert_eq!(marked.len(), self.slot_len);
        let mut reclaimed = 0;
        for index in 0..self.slot_len {
            if marked[index] {
                if let Some(value) = self.slot_mut(index).and_then(|slot| slot.value.as_mut()) {
                    visit_retained(value);
                }
                continue;
            }
            let Some(slot) = self.slot(index) else {
                continue;
            };
            if slot.value.is_none() {
                continue;
            }
            let generation = slot.generation;
            let index = u32::try_from(index).expect("arena index is bounded by insertion");
            if self.remove(index, generation).is_some() {
                reclaimed += 1;
            }
        }
        reclaimed
    }
}

/// Owner of all mutable datum and list identities.
#[derive(Default)]
pub struct ValueHeap {
    datums: Arena<Datum>,
    lists: Arena<DmList>,
    datum_layouts: DatumLayoutCache,
}

/// Pointer-free logical value stored in a ready-world heap image.
///
/// Text and identifier storage is owned so an image does not retain process
/// addresses. Datum and list references preserve their stable slot and
/// generation identities.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum HeapSnapshotValue {
    /// DM `null`.
    Null,
    /// Exact binary32 number bits.
    Number(u32),
    /// Immutable text content.
    Text(String),
    /// Project-relative file/resource path.
    File(String),
    /// Canonical type path.
    TypePath(String),
    /// Modified type path and evaluated overrides.
    ModifiedTypePath {
        /// Canonical base path.
        base: String,
        /// Overrides in source order.
        overrides: Vec<(String, HeapSnapshotValue)>,
    },
    /// Stable datum identity.
    Datum {
        /// Stable arena slot.
        index: u32,
        /// Slot generation.
        generation: u32,
    },
    /// Stable list identity.
    List {
        /// Stable arena slot.
        index: u32,
        /// Slot generation.
        generation: u32,
    },
}

/// One materialized datum in a ready-world heap image.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct HeapSnapshotDatum {
    /// Canonical runtime type.
    pub type_path: String,
    /// Materialized fields in declaration/insertion order.
    pub fields: Vec<(String, HeapSnapshotValue)>,
}

/// One entry in a list's unified iteration order.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum HeapSnapshotListEntry {
    /// Positional value.
    Positional(HeapSnapshotValue),
    /// Associative key and value.
    Associative(HeapSnapshotValue, HeapSnapshotValue),
}

/// One arena slot in a ready-world heap image.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct HeapSnapshotSlot<T> {
    /// Generation used to reject stale handles.
    pub generation: u32,
    /// Live payload, or `None` for a reusable vacant slot.
    pub value: Option<T>,
}

/// Pointer-free snapshot of the mutable value heap.
///
/// This is the first persistent section of a ready-world image. It retains
/// vacant-slot generations and free-list order so allocations after restore
/// produce the same logical identities as uninterrupted execution.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ValueHeapSnapshot {
    /// Datum arena slots in stable index order.
    pub datums: Vec<HeapSnapshotSlot<HeapSnapshotDatum>>,
    /// Datum free-list pop order.
    pub datum_free: Vec<u32>,
    /// List arena slots in stable index order.
    pub lists: Vec<HeapSnapshotSlot<Vec<HeapSnapshotListEntry>>>,
    /// List free-list pop order.
    pub list_free: Vec<u32>,
}

/// Results and storage telemetry from one combined datum/list collection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeapCollectionStats {
    /// Unreachable datum identities reclaimed by this collection.
    pub reclaimed_datums: usize,
    /// Unreachable list identities reclaimed by this collection.
    pub reclaimed_lists: usize,
    /// Live list backing storage measured during the existing sweep pass.
    pub list_storage: ListStorageStats,
    /// Live datum backing storage measured during the existing sweep pass.
    pub datum_storage: DatumStorageStats,
    /// Datum arena allocation state after sweeping.
    pub datum_arena: ArenaStats,
    /// List arena allocation state after sweeping.
    pub list_arena: ArenaStats,
}

const PARALLEL_ROOT_VALIDATION_THRESHOLD: usize = 32_768;

/// Validates independent GC roots against the immutable heap view. Mark graph
/// mutation and sweeping remain on the collector thread; only this read-only,
/// embarrassingly parallel phase is distributed.
fn validate_gc_roots_parallel<T, F>(roots: &[T], is_live: F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T) -> bool + Sync,
{
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    if workers <= 1 || roots.len() < PARALLEL_ROOT_VALIDATION_THRESHOLD {
        return roots
            .iter()
            .copied()
            .filter(|root| is_live(*root))
            .collect();
    }
    // Indexed parallel iteration retains source order when collected into a
    // Vec, so the subsequent graph walk and sweep remain deterministic.
    roots
        .par_iter()
        .copied()
        .filter(|root| is_live(*root))
        .collect()
}

impl ValueHeap {
    /// Creates an empty heap.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Captures all logical heap state without retaining process pointers or
    /// derived lookup indexes.
    ///
    /// The returned representation is suitable for a versioned ready-world
    /// image encoder. Derived datum/list indexes and layout sharing are rebuilt
    /// on restore instead of being persisted.
    #[must_use]
    pub fn snapshot(&self) -> ValueHeapSnapshot {
        let datums = (0..self.datums.slot_len())
            .map(|index| {
                let slot = self
                    .datums
                    .slot(index)
                    .expect("datum snapshot index addresses an allocated slot");
                HeapSnapshotSlot {
                    generation: slot.generation,
                    value: slot.value.as_ref().map(|datum| HeapSnapshotDatum {
                        type_path: datum.type_path().as_str().to_owned(),
                        fields: datum
                            .fields()
                            .map(|(name, value)| {
                                (name.as_str().to_owned(), HeapSnapshotValue::from(value))
                            })
                            .collect(),
                    }),
                }
            })
            .collect();
        let lists = (0..self.lists.slot_len())
            .map(|index| {
                let slot = self
                    .lists
                    .slot(index)
                    .expect("list snapshot index addresses an allocated slot");
                HeapSnapshotSlot {
                    generation: slot.generation,
                    value: slot.value.as_ref().map(snapshot_list_entries),
                }
            })
            .collect();
        ValueHeapSnapshot {
            datums,
            datum_free: self.datums.free.clone(),
            lists,
            list_free: self.lists.free.clone(),
        }
    }

    /// Restores a heap snapshot while preserving every stable handle.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::CorruptHeapSnapshot`] for malformed identifiers,
    /// inconsistent free lists, or corrupt list records.
    pub fn from_snapshot(snapshot: ValueHeapSnapshot) -> Result<Self, ValueError> {
        let datum_slots = snapshot
            .datums
            .into_iter()
            .map(|slot| {
                let value = slot.value.map(restore_snapshot_datum).transpose()?;
                Ok((slot.generation, value))
            })
            .collect::<Result<Vec<_>, ValueError>>()?;
        let list_slots = snapshot
            .lists
            .into_iter()
            .map(|slot| {
                let value = slot.value.map(restore_list_entries).transpose()?;
                Ok((slot.generation, value))
            })
            .collect::<Result<Vec<_>, ValueError>>()?;
        let datums = arena_from_snapshot_slots(datum_slots, snapshot.datum_free, "datum")?;
        let lists = arena_from_snapshot_slots(list_slots, snapshot.list_free, "list")?;
        let mut heap = Self {
            datums,
            lists,
            datum_layouts: DatumLayoutCache::default(),
        };
        heap.compact_restored_datum_layouts();
        Ok(heap)
    }

    fn compact_restored_datum_layouts(&mut self) {
        // Recreate field-name/index sharing without changing logical state.
        for index in 0..self.datums.slot_len() {
            let Some(datum) = self
                .datums
                .slot_mut(index)
                .and_then(|slot| slot.value.as_mut())
            else {
                continue;
            };
            self.datum_layouts.compact(datum);
        }
    }

    /// Returns the number of currently live datum identities.
    #[must_use]
    pub fn live_datum_count(&self) -> usize {
        self.datums.len()
    }

    /// Returns the number of currently live list identities.
    #[must_use]
    pub fn live_list_count(&self) -> usize {
        self.lists.len()
    }

    /// Returns constant-time datum arena allocation telemetry.
    #[must_use]
    pub fn datum_arena_stats(&self) -> ArenaStats {
        self.datums.stats()
    }

    /// Returns constant-time list arena allocation telemetry.
    #[must_use]
    pub fn list_arena_stats(&self) -> ArenaStats {
        self.lists.stats()
    }

    /// Reclaims every list that is unreachable from live datum fields or the
    /// additional runtime roots supplied by the caller.
    ///
    /// DM lists may recursively contain other lists, including through
    /// associative keys and values. This performs a complete transitive mark
    /// before invalidating unreachable handles, so aliases retained by a live
    /// datum or runtime frame preserve their identity.
    pub fn collect_unreachable_lists(&mut self, roots: &[Value]) -> usize {
        fn enqueue(value: &Value, pending: &mut Vec<ListId>) {
            match value {
                Value::List(list) => pending.push(*list),
                Value::ModifiedTypePath(path) => {
                    for (_, value) in path.overrides() {
                        enqueue(value, pending);
                    }
                }
                Value::Null
                | Value::Number(_)
                | Value::Text(_)
                | Value::File(_)
                | Value::TypePath(_)
                | Value::Datum(_) => {}
            }
        }

        let mut pending = Vec::new();
        for (_, datum) in self.datums() {
            for (_, value) in datum.fields() {
                enqueue(value, &mut pending);
            }
        }
        for value in roots {
            enqueue(value, &mut pending);
        }

        let mut marked = HashSet::new();
        while let Some(list) = pending.pop() {
            if !marked.insert(list) {
                continue;
            }
            let Ok(values) = self.list(list) else {
                continue;
            };
            for (_, value) in values.positions() {
                enqueue(value, &mut pending);
            }
            for (_, value) in values.associations() {
                enqueue(value, &mut pending);
            }
        }

        let unreachable = self
            .lists
            .iter()
            .map(|(index, generation, _)| ListId { index, generation })
            .filter(|list| !marked.contains(list))
            .collect::<Vec<_>>();
        for list in &unreachable {
            let removed = self.lists.remove(list.index, list.generation);
            debug_assert!(removed.is_some());
        }
        unreachable.len()
    }

    /// Reclaims every datum and list unreachable from the supplied runtime roots.
    ///
    /// Datum fields and list entries form one object graph, including cycles.
    /// This marks both identity kinds transitively before invalidating any
    /// handles, matching DM's collection of unreferenced datum/list cycles.
    /// Returns the number of reclaimed datums followed by reclaimed lists.
    pub fn collect_unreachable_values(&mut self, roots: &[Value]) -> (usize, usize) {
        fn collect_root_ids(
            value: &Value,
            datum_roots: &mut Vec<DatumId>,
            list_roots: &mut Vec<ListId>,
        ) {
            match value {
                Value::Datum(datum) => datum_roots.push(*datum),
                Value::List(list) => list_roots.push(*list),
                Value::ModifiedTypePath(path) => {
                    for (_, value) in path.overrides() {
                        collect_root_ids(value, datum_roots, list_roots);
                    }
                }
                Value::Null
                | Value::Number(_)
                | Value::Text(_)
                | Value::File(_)
                | Value::TypePath(_) => {}
            }
        }

        let mut datum_roots = Vec::new();
        let mut list_roots = Vec::new();
        for value in roots {
            collect_root_ids(value, &mut datum_roots, &mut list_roots);
        }
        self.collect_unreachable_values_from_ids(&datum_roots, &list_roots)
    }

    /// Reclaims every datum and list unreachable from compact identity roots.
    ///
    /// This variant avoids materializing a full [`Value`] for each root in
    /// large worlds. Dense arena-index mark vectors also avoid hash-table
    /// capacity spikes while preserving stable-generation validation.
    pub fn collect_unreachable_values_from_ids(
        &mut self,
        datum_roots: &[DatumId],
        list_roots: &[ListId],
    ) -> (usize, usize) {
        let stats = self.collect_unreachable_values_from_ids_with_stats(datum_roots, list_roots);
        (stats.reclaimed_datums, stats.reclaimed_lists)
    }

    /// Reclaims unreachable identities and reports storage observed during the
    /// same sweep, avoiding a second full heap walk for diagnostics.
    pub fn collect_unreachable_values_from_ids_with_stats(
        &mut self,
        datum_roots: &[DatumId],
        list_roots: &[ListId],
    ) -> HeapCollectionStats {
        #[derive(Clone, Copy)]
        enum Pending {
            Datum(DatumId),
            List(ListId),
        }

        fn enqueue_value(
            heap: &ValueHeap,
            value: &Value,
            pending: &mut Vec<Pending>,
            marked_datums: &mut [bool],
            marked_lists: &mut [bool],
        ) {
            match value {
                Value::Datum(datum) => {
                    enqueue_datum(heap, *datum, pending, marked_datums, marked_lists)
                }
                Value::List(list) => {
                    enqueue_list(heap, *list, pending, marked_datums, marked_lists)
                }
                Value::ModifiedTypePath(path) => {
                    for (_, value) in path.overrides() {
                        enqueue_value(heap, value, pending, marked_datums, marked_lists);
                    }
                }
                Value::Null
                | Value::Number(_)
                | Value::Text(_)
                | Value::File(_)
                | Value::TypePath(_) => {}
            }
        }

        fn enqueue_datum(
            heap: &ValueHeap,
            datum: DatumId,
            pending: &mut Vec<Pending>,
            marked_datums: &mut [bool],
            _marked_lists: &mut [bool],
        ) {
            let index = datum.index() as usize;
            if index >= marked_datums.len() || marked_datums[index] || heap.datum(datum).is_err() {
                return;
            }
            marked_datums[index] = true;
            pending.push(Pending::Datum(datum));
        }

        fn enqueue_list(
            heap: &ValueHeap,
            list: ListId,
            pending: &mut Vec<Pending>,
            _marked_datums: &mut [bool],
            marked_lists: &mut [bool],
        ) {
            let index = list.index() as usize;
            if index >= marked_lists.len() || marked_lists[index] || heap.list(list).is_err() {
                return;
            }
            marked_lists[index] = true;
            pending.push(Pending::List(list));
        }

        let mut pending = Vec::new();
        let mut marked_datums = vec![false; self.datums.slot_len()];
        let mut marked_lists = vec![false; self.lists.slot_len()];
        let datum_roots =
            validate_gc_roots_parallel(datum_roots, |datum| self.datum(datum).is_ok());
        let list_roots = validate_gc_roots_parallel(list_roots, |list| self.list(list).is_ok());
        for datum in &datum_roots {
            let index = datum.index() as usize;
            if !marked_datums[index] {
                marked_datums[index] = true;
                pending.push(Pending::Datum(*datum));
            }
        }
        for list in &list_roots {
            let index = list.index() as usize;
            if !marked_lists[index] {
                marked_lists[index] = true;
                pending.push(Pending::List(*list));
            }
        }
        while let Some(value) = pending.pop() {
            match value {
                Pending::Datum(datum) => {
                    let Ok(datum) = self.datum(datum) else {
                        continue;
                    };
                    for (_, value) in datum.fields() {
                        enqueue_value(
                            self,
                            value,
                            &mut pending,
                            &mut marked_datums,
                            &mut marked_lists,
                        );
                    }
                }
                Pending::List(list) => {
                    let Ok(list) = self.list(list) else {
                        continue;
                    };
                    for (_, value) in list.positions() {
                        enqueue_value(
                            self,
                            value,
                            &mut pending,
                            &mut marked_datums,
                            &mut marked_lists,
                        );
                    }
                    for (_, value) in list.associations() {
                        enqueue_value(
                            self,
                            value,
                            &mut pending,
                            &mut marked_datums,
                            &mut marked_lists,
                        );
                    }
                }
            }
        }

        let mut list_storage = ListStorageStats::default();
        let reclaimed_lists = self.lists.sweep_unmarked_with(&marked_lists, |list| {
            list.compact_and_measure_for_gc(&mut list_storage);
        });
        let mut datum_storage = DatumStorageStats::default();
        let mut field_indexes = FieldIndexInterner::default();
        let reclaimed_datums = self.datums.sweep_unmarked_with(&marked_datums, |datum| {
            datum.compact_and_measure_for_gc(&mut datum_storage, &mut field_indexes);
        });
        datum_storage.shared_field_name_layouts = field_indexes.shared_name_layouts();
        datum_storage.shared_field_name_physical_slots = field_indexes.shared_name_slots();
        datum_storage.shared_field_name_bytes_saved = datum_storage
            .shared_field_name_logical_slots
            .saturating_sub(datum_storage.shared_field_name_physical_slots)
            .saturating_mul(std::mem::size_of::<FieldName>());
        HeapCollectionStats {
            reclaimed_datums,
            reclaimed_lists,
            list_storage,
            datum_storage,
            datum_arena: self.datums.stats(),
            list_arena: self.lists.stats(),
        }
    }

    /// Allocates an empty mutable list.
    pub fn allocate_list(&mut self) -> ListId {
        let (index, generation) = self.lists.insert(DmList::default());
        ListId { index, generation }
    }

    /// Returns a live list.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleList`] for stale or unknown identities.
    pub fn list(&self, id: ListId) -> Result<&DmList, ValueError> {
        self.lists
            .get(id.index, id.generation)
            .ok_or(ValueError::StaleList(id))
    }

    /// Returns a live mutable list.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleList`] for stale or unknown identities.
    pub fn list_mut(&mut self, id: ListId) -> Result<&mut DmList, ValueError> {
        self.lists
            .get_mut(id.index, id.generation)
            .ok_or(ValueError::StaleList(id))
    }

    /// Deallocates a list and invalidates every alias to its identity.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleList`] for stale or unknown identities.
    pub fn destroy_list(&mut self, id: ListId) -> Result<DmList, ValueError> {
        self.lists
            .remove(id.index, id.generation)
            .ok_or(ValueError::StaleList(id))
    }

    /// Creates a shallow list copy with a distinct reference identity.
    ///
    /// Nested datum/list values retain their existing handles.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleList`] when `source` is stale.
    pub fn copy_list(&mut self, source: ListId) -> Result<ListId, ValueError> {
        let copy = self.list(source)?.clone();
        let (index, generation) = self.lists.insert(copy);
        Ok(ListId { index, generation })
    }

    /// Allocates a datum with its runtime type.
    pub fn allocate_datum(&mut self, type_path: TypePath) -> DatumId {
        let (index, generation) = self.datums.insert(Datum {
            type_path,
            fields: DatumFields::default(),
            field_index: None,
        });
        DatumId { index, generation }
    }

    /// Allocates a datum after applying resolved default layers in order.
    ///
    /// The caller owns inheritance resolution and should pass layers from the
    /// oldest ancestor through the concrete type. Layer paths are retained for
    /// diagnostics by their owners but are intentionally not coupled to a
    /// compiler object-tree identity. Default values are cloned shallowly.
    pub fn allocate_datum_with_defaults(
        &mut self,
        type_path: TypePath,
        defaults: &[DatumDefaults],
    ) -> DatumId {
        let mut datum = Datum {
            type_path,
            fields: DatumFields::default(),
            field_index: None,
        };
        for layer in defaults {
            datum.apply_defaults(layer);
        }
        self.datum_layouts.compact(&mut datum);
        let (index, generation) = self.datums.insert(datum);
        DatumId { index, generation }
    }

    /// Shares an initialized datum's immutable field-name layout while
    /// retaining a distinct mutable value vector for the datum.
    ///
    /// Call this after engine-owned defaults and map overrides have been
    /// materialized. Later insertion or deletion detaches only that datum;
    /// ordinary value mutation remains compact.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleDatum`] for a stale identity.
    pub fn compact_datum_layout(&mut self, id: DatumId) -> Result<(), ValueError> {
        let datum = self
            .datums
            .get_mut(id.index, id.generation)
            .ok_or(ValueError::StaleDatum(id))?;
        self.datum_layouts.compact(datum);
        Ok(())
    }

    /// Returns a live datum.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleDatum`] for stale or unknown identities.
    pub fn datum(&self, id: DatumId) -> Result<&Datum, ValueError> {
        self.datums
            .get(id.index, id.generation)
            .ok_or(ValueError::StaleDatum(id))
    }

    /// Returns a live mutable datum.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleDatum`] for stale or unknown identities.
    pub fn datum_mut(&mut self, id: DatumId) -> Result<&mut Datum, ValueError> {
        self.datums
            .get_mut(id.index, id.generation)
            .ok_or(ValueError::StaleDatum(id))
    }

    /// Reads a field from a live datum.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleDatum`] for a stale identity or
    /// [`ValueError::MissingField`] for an absent field.
    pub fn datum_field(&self, id: DatumId, name: &FieldName) -> Result<&Value, ValueError> {
        self.datum(id)?.field(name)
    }

    /// Reads an optionally materialized field from a live datum.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleDatum`] for a stale identity. An absent field
    /// returns `Ok(None)` without constructing a [`ValueError::MissingField`].
    pub fn datum_field_optional(
        &self,
        id: DatumId,
        name: &FieldName,
    ) -> Result<Option<&Value>, ValueError> {
        Ok(self.datum(id)?.field_optional(name))
    }

    /// Replaces the runtime type of a live datum while preserving its stable
    /// identity. Engine-owned turf replacement uses this for one map cell.
    pub fn set_datum_type_path(
        &mut self,
        id: DatumId,
        type_path: TypePath,
    ) -> Result<TypePath, ValueError> {
        let datum = self.datum_mut(id)?;
        Ok(std::mem::replace(&mut datum.type_path, type_path))
    }

    /// Iterates the materialized fields of a live datum in declaration order.
    pub fn datum_fields(
        &self,
        id: DatumId,
    ) -> Result<impl Iterator<Item = (&FieldName, &Value)>, ValueError> {
        Ok(self.datum(id)?.fields())
    }

    /// Inserts or updates a field on a live datum.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleDatum`] for a stale identity.
    pub fn set_datum_field(
        &mut self,
        id: DatumId,
        name: FieldName,
        value: Value,
    ) -> Result<Option<Value>, ValueError> {
        Ok(self.datum_mut(id)?.set_field(name, value))
    }

    /// Deletes a field from a live datum.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleDatum`] for a stale identity. A live datum
    /// with no such field returns `Ok(None)`.
    pub fn delete_datum_field(
        &mut self,
        id: DatumId,
        name: &FieldName,
    ) -> Result<Option<Value>, ValueError> {
        Ok(self.datum_mut(id)?.delete_field(name))
    }

    /// Iterates live datums in stable arena-slot order for snapshots.
    ///
    /// Deletions leave holes and slot reuse receives a new generation, so a
    /// snapshot consumer must retain the complete [`DatumId`].
    pub fn datums(&self) -> impl Iterator<Item = (DatumId, &Datum)> {
        self.datums
            .iter()
            .map(|(index, generation, datum)| (DatumId { index, generation }, datum))
    }

    /// Returns the live datum identity occupying an arena slot, if any.
    ///
    /// BYOND reference text encodes the stable object slot. Resolving that
    /// text needs the slot's current generation without scanning every live
    /// datum in large worlds.
    #[must_use]
    pub fn datum_id_at_index(&self, index: u32) -> Option<DatumId> {
        self.datums
            .live_generation(index)
            .map(|generation| DatumId { index, generation })
    }

    /// Returns the live list identity occupying an arena slot, if any.
    #[must_use]
    pub fn list_id_at_index(&self, index: u32) -> Option<ListId> {
        self.lists
            .live_generation(index)
            .map(|generation| ListId { index, generation })
    }

    /// Deallocates a datum and invalidates every alias to its identity.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::StaleDatum`] for stale or unknown identities.
    pub fn destroy_datum(&mut self, id: DatumId) -> Result<Datum, ValueError> {
        self.datums
            .remove(id.index, id.generation)
            .ok_or(ValueError::StaleDatum(id))
    }

    /// Evaluates DM truth, treating deleted heap references as null.
    ///
    /// Null, numeric zero (including signed zero), and empty text are false.
    /// Live datum/list references and type paths are true. Deleted datum/list
    /// references are false, matching BYOND's null-like hard-delete behavior.
    /// NaN is currently true because it is nonzero; this requires differential
    /// confirmation.
    pub fn truthy(&self, value: &Value) -> Result<bool, ValueError> {
        match value {
            Value::Null => Ok(false),
            Value::Number(number) => Ok(number.to_f32() != 0.0),
            Value::Text(text) => Ok(!text.is_empty()),
            Value::File(_) => Ok(true),
            Value::TypePath(_) => Ok(true),
            Value::ModifiedTypePath(_) => Ok(true),
            Value::Datum(id) => Ok(self.datum(*id).is_ok()),
            Value::List(id) => Ok(self.list(*id).is_ok()),
        }
    }
}

impl From<&Value> for HeapSnapshotValue {
    fn from(value: &Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Number(number) => Self::Number(number.bits()),
            Value::Text(text) => Self::Text(text.to_string()),
            Value::File(path) => Self::File(path.to_string()),
            Value::TypePath(path) => Self::TypePath(path.as_str().to_owned()),
            Value::ModifiedTypePath(path) => Self::ModifiedTypePath {
                base: path.base().as_str().to_owned(),
                overrides: path
                    .overrides()
                    .iter()
                    .map(|(name, value)| (name.as_str().to_owned(), HeapSnapshotValue::from(value)))
                    .collect(),
            },
            Value::Datum(datum) => Self::Datum {
                index: datum.index(),
                generation: datum.generation(),
            },
            Value::List(list) => Self::List {
                index: list.index(),
                generation: list.generation(),
            },
        }
    }
}

impl HeapSnapshotValue {
    /// Reconstructs the runtime value represented by this pointer-free record.
    ///
    /// # Errors
    ///
    /// Returns a value validation error for malformed field or type names.
    pub fn into_value(self) -> Result<Value, ValueError> {
        Ok(match self {
            Self::Null => Value::Null,
            Self::Number(bits) => Value::Number(DmNumberBits::from_f32(f32::from_bits(bits))),
            Self::Text(text) => Value::text(text),
            Self::File(path) => Value::file(path),
            Self::TypePath(path) => Value::TypePath(TypePath::parse(&path)?),
            Self::ModifiedTypePath { base, overrides } => {
                let overrides = overrides
                    .into_iter()
                    .map(|(name, value)| Ok((FieldName::parse(&name)?, value.into_value()?)))
                    .collect::<Result<Vec<_>, ValueError>>()?;
                Value::ModifiedTypePath(Arc::new(ModifiedTypePath::new(
                    TypePath::parse(&base)?,
                    overrides,
                )))
            }
            Self::Datum { index, generation } => Value::Datum(DatumId { index, generation }),
            Self::List { index, generation } => Value::List(ListId { index, generation }),
        })
    }
}

fn restore_snapshot_datum(datum: HeapSnapshotDatum) -> Result<Datum, ValueError> {
    let type_path = TypePath::parse(&datum.type_path)?;
    let mut fields = Vec::with_capacity(datum.fields.len());
    for (name, value) in datum.fields {
        fields.push((FieldName::parse(&name)?, value.into_value()?));
    }
    let mut datum = Datum {
        type_path,
        fields: DatumFields::Owned(fields),
        field_index: None,
    };
    if datum.fields.len() >= DATUM_FIELD_INDEX_THRESHOLD {
        datum.build_field_index();
    }
    Ok(datum)
}

fn snapshot_list_entries(list: &DmList) -> Vec<HeapSnapshotListEntry> {
    let Some(storage) = list.storage.as_deref() else {
        return Vec::new();
    };
    storage.order[storage.prefix_head..]
        .iter()
        .map(|entry| {
            if let Some(index) = entry.positional_index() {
                HeapSnapshotListEntry::Positional(HeapSnapshotValue::from(
                    &storage.positional[index],
                ))
            } else {
                let index = entry
                    .associative_index()
                    .expect("live list order entry is valid");
                let (key, value) = &storage.associative[index];
                HeapSnapshotListEntry::Associative(
                    HeapSnapshotValue::from(key),
                    HeapSnapshotValue::from(value),
                )
            }
        })
        .collect()
}

fn restore_list_entries(entries: Vec<HeapSnapshotListEntry>) -> Result<DmList, ValueError> {
    if entries.is_empty() {
        return Ok(DmList::default());
    }
    let mut positional = Vec::new();
    let mut associative = Vec::new();
    let mut order = Vec::with_capacity(entries.len());
    for entry in entries {
        match entry {
            HeapSnapshotListEntry::Positional(value) => {
                order.push(ListOrder::positional(positional.len()));
                positional.push(value.into_value()?);
            }
            HeapSnapshotListEntry::Associative(key, value) => {
                order.push(ListOrder::associative(associative.len()));
                associative.push((key.into_value()?, value.into_value()?));
            }
        }
    }
    let mut list = DmList {
        storage: Some(Arc::new(DmListStorage {
            positional,
            associative,
            order,
            associative_index: None,
            positional_remove_index: None,
            prefix_head: 0,
        })),
    };
    list.rebuild_associative_index();
    Ok(list)
}

fn arena_from_snapshot_slots<T>(
    slots: Vec<(u32, Option<T>)>,
    free: Vec<u32>,
    kind: &str,
) -> Result<Arena<T>, ValueError> {
    let mut free_seen = vec![false; slots.len()];
    for &index in &free {
        let index = index as usize;
        if index >= slots.len() || free_seen[index] || slots[index].1.is_some() {
            return Err(ValueError::CorruptHeapSnapshot(format!(
                "{kind} free list contains invalid slot {index}"
            )));
        }
        free_seen[index] = true;
    }
    for (index, (generation, value)) in slots.iter().enumerate() {
        if value.is_none() && *generation != u32::MAX && !free_seen[index] {
            return Err(ValueError::CorruptHeapSnapshot(format!(
                "{kind} vacant slot {index} is absent from its free list"
            )));
        }
    }
    let live = slots.iter().filter(|(_, value)| value.is_some()).count();
    let slot_len = slots.len();
    let mut chunks = Vec::with_capacity(slot_len.div_ceil(ARENA_CHUNK_SLOTS));
    for chunk in slots
        .into_iter()
        .collect::<Vec<_>>()
        .chunks_mut(ARENA_CHUNK_SLOTS)
    {
        let mut values = Vec::with_capacity(ARENA_CHUNK_SLOTS);
        values.extend(chunk.iter_mut().map(|(generation, value)| Slot {
            generation: *generation,
            value: value.take(),
        }));
        chunks.push(values);
    }
    Ok(Arena {
        chunks,
        slot_len,
        free,
        live,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(value: &str) -> Value {
        Value::text(value)
    }

    fn field(value: &str) -> FieldName {
        FieldName::parse(value).unwrap()
    }

    fn index_number(index: usize) -> Value {
        Value::number(f32::from(u16::try_from(index).unwrap()))
    }

    fn index_f32(index: usize) -> f32 {
        f32::from(u16::try_from(index).unwrap())
    }

    #[test]
    fn type_paths_are_canonical_and_validated() {
        let path = TypePath::parse("/obj/item/tool").unwrap();
        assert_eq!(path.as_str(), "/obj/item/tool");
        assert_eq!(path.to_string(), "/obj/item/tool");

        for invalid in ["obj/item", "/", "/obj/", "/obj//item", ""] {
            assert_eq!(
                TypePath::parse(invalid),
                Err(ValueError::InvalidTypePath(invalid.to_owned()))
            );
        }
    }

    #[test]
    fn field_names_are_canonical_identifiers() {
        for valid in ["name", "icon_state", "_private", "field2"] {
            let name = FieldName::parse(valid).unwrap();
            assert_eq!(name.as_str(), valid);
            assert_eq!(name.to_string(), valid);
        }
        for invalid in ["", "2field", "field-name", "field/name", "naïve"] {
            assert_eq!(
                FieldName::parse(invalid),
                Err(ValueError::InvalidFieldName(invalid.to_owned()))
            );
        }
    }

    #[test]
    fn datum_fields_support_ordered_get_set_and_delete() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/example").unwrap());
        let name = field("name");
        let count = field("count");

        assert!(
            heap.set_datum_field(datum, name.clone(), text("first"))
                .unwrap()
                .is_none()
        );
        heap.set_datum_field(datum, count.clone(), Value::number(2.0))
            .unwrap();
        let old = heap
            .set_datum_field(datum, name.clone(), text("updated"))
            .unwrap()
            .unwrap();
        assert!(old.semantic_eq(&text("first")));
        assert!(
            heap.datum_field(datum, &name)
                .unwrap()
                .semantic_eq(&text("updated"))
        );
        let names: Vec<&str> = heap
            .datum(datum)
            .unwrap()
            .fields()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names, ["name", "count"]);

        assert!(
            heap.delete_datum_field(datum, &name)
                .unwrap()
                .unwrap()
                .semantic_eq(&text("updated"))
        );
        assert_eq!(
            heap.datum_field(datum, &name),
            Err(ValueError::MissingField(name))
        );
        assert_eq!(heap.datum(datum).unwrap().field_len(), 1);
    }

    #[test]
    fn optional_datum_field_distinguishes_sparse_slots_from_stale_handles() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/example").unwrap());
        let present = field("present");
        let absent = field("absent");
        heap.set_datum_field(datum, present.clone(), Value::number(7.0))
            .unwrap();

        assert_eq!(
            heap.datum_field_optional(datum, &present)
                .unwrap()
                .and_then(Value::as_number),
            Some(7.0)
        );
        assert_eq!(heap.datum_field_optional(datum, &absent), Ok(None));

        heap.destroy_datum(datum).unwrap();
        assert_eq!(
            heap.datum_field_optional(datum, &present),
            Err(ValueError::StaleDatum(datum))
        );
    }

    #[test]
    fn datum_field_index_appears_exactly_at_threshold_without_changing_order() {
        assert_eq!(
            std::mem::size_of::<Datum>(),
            std::mem::size_of::<TypePath>()
                + std::mem::size_of::<DatumFields>()
                + std::mem::size_of::<Option<Arc<HashMap<FieldName, usize>>>>(),
        );
        assert_eq!(
            std::mem::size_of::<Option<Box<HashMap<FieldName, usize>>>>(),
            std::mem::size_of::<usize>(),
        );

        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/indexed").unwrap());

        for index in 0..DATUM_FIELD_INDEX_THRESHOLD - 1 {
            heap.set_datum_field(
                datum,
                field(&format!("field_{index:02}")),
                index_number(index),
            )
            .unwrap();
        }
        assert!(heap.datum(datum).unwrap().field_index.is_none());

        let last = DATUM_FIELD_INDEX_THRESHOLD - 1;
        heap.set_datum_field(
            datum,
            field(&format!("field_{last:02}")),
            index_number(last),
        )
        .unwrap();
        let record = heap.datum(datum).unwrap();
        assert_eq!(
            record.field_index.as_ref().unwrap().len(),
            record.field_len()
        );
        for index in 0..DATUM_FIELD_INDEX_THRESHOLD {
            assert_eq!(
                record
                    .field(&field(&format!("field_{index:02}")))
                    .unwrap()
                    .as_number(),
                Some(index_f32(index))
            );
        }
        let names: Vec<String> = record
            .fields()
            .map(|(name, _)| name.as_str().to_owned())
            .collect();
        let expected: Vec<String> = (0..DATUM_FIELD_INDEX_THRESHOLD)
            .map(|index| format!("field_{index:02}"))
            .collect();
        assert_eq!(names, expected);

        let updated_name = field("field_00");
        assert_eq!(
            heap.set_datum_field(datum, updated_name.clone(), Value::number(100.0))
                .unwrap()
                .unwrap()
                .as_number(),
            Some(0.0)
        );
        let record = heap.datum(datum).unwrap();
        assert_eq!(
            record.field(&updated_name).unwrap().as_number(),
            Some(100.0)
        );
        assert_eq!(record.fields().next().unwrap().0, &updated_name);
        assert_eq!(
            record.field(&field("absent")),
            Err(ValueError::MissingField(field("absent")))
        );
    }

    #[test]
    fn datum_field_index_reindexes_after_delete_and_drops_below_threshold() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/indexed").unwrap());
        let field_count = DATUM_FIELD_INDEX_THRESHOLD + 3;
        for index in 0..field_count {
            heap.set_datum_field(
                datum,
                field(&format!("field_{index:02}")),
                index_number(index),
            )
            .unwrap();
        }

        let deleted = field("field_01");
        assert_eq!(
            heap.delete_datum_field(datum, &deleted)
                .unwrap()
                .unwrap()
                .as_number(),
            Some(1.0)
        );
        let record = heap.datum(datum).unwrap();
        let field_index = record.field_index.as_ref().unwrap();
        assert_eq!(field_index.len(), field_count - 1);
        for (position, (name, value)) in record.fields().enumerate() {
            assert_eq!(field_index.get(name), Some(&position));
            assert_eq!(record.field(name).unwrap(), value);
        }
        assert_eq!(
            record.field(&deleted),
            Err(ValueError::MissingField(deleted))
        );

        for index in [0, 2, 3] {
            heap.delete_datum_field(datum, &field(&format!("field_{index:02}")))
                .unwrap();
        }
        let record = heap.datum(datum).unwrap();
        assert_eq!(record.field_len(), DATUM_FIELD_INDEX_THRESHOLD - 1);
        assert!(record.field_index.is_none());
        for (name, value) in record.fields() {
            assert_eq!(record.field(name).unwrap(), value);
        }
    }

    #[test]
    fn indexed_datum_defaults_and_clones_preserve_layer_and_value_semantics() {
        let mut parent = DatumDefaults::new(TypePath::parse("/datum").unwrap());
        for index in 0..DATUM_FIELD_INDEX_THRESHOLD - 4 {
            parent.set(field(&format!("field_{index:02}")), index_number(index));
        }
        let mut child = DatumDefaults::new(TypePath::parse("/datum/child").unwrap());
        child.set(field("field_03"), Value::number(303.0));
        for index in DATUM_FIELD_INDEX_THRESHOLD - 4..DATUM_FIELD_INDEX_THRESHOLD + 2 {
            child.set(field(&format!("field_{index:02}")), index_number(index));
        }

        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum_with_defaults(
            TypePath::parse("/datum/child/example").unwrap(),
            &[parent, child],
        );
        let record = heap.datum(datum).unwrap();
        assert_eq!(record.field_len(), DATUM_FIELD_INDEX_THRESHOLD + 2);
        assert!(record.field_index.is_some());
        assert_eq!(
            record.field(&field("field_03")).unwrap().as_number(),
            Some(303.0)
        );
        let expected_names: Vec<String> = (0..DATUM_FIELD_INDEX_THRESHOLD + 2)
            .map(|index| format!("field_{index:02}"))
            .collect();
        assert_eq!(
            record
                .fields()
                .map(|(name, _)| name.as_str().to_owned())
                .collect::<Vec<_>>(),
            expected_names
        );

        let mut cloned = record.clone();
        assert_eq!(cloned, *record);
        assert!(cloned.field_index.is_some());
        assert_eq!(
            cloned
                .set_field(field("field_03"), Value::number(404.0))
                .unwrap()
                .as_number(),
            Some(303.0)
        );
        assert_eq!(
            cloned.field(&field("field_03")).unwrap().as_number(),
            Some(404.0)
        );
        assert_eq!(
            record.field(&field("field_03")).unwrap().as_number(),
            Some(303.0)
        );
    }

    #[test]
    fn resolved_default_layers_apply_parent_first_with_stable_order() {
        let mut parent = DatumDefaults::new(TypePath::parse("/datum").unwrap());
        parent.set(field("name"), text("parent"));
        parent.set(field("health"), Value::number(10.0));
        let mut child = DatumDefaults::new(TypePath::parse("/datum/child").unwrap());
        child.set(field("name"), text("child"));
        child.set(field("speed"), Value::number(2.0));

        let mut heap = ValueHeap::new();
        let runtime_type = TypePath::parse("/datum/child/grandchild").unwrap();
        let datum = heap
            .allocate_datum_with_defaults(runtime_type.clone(), &[parent.clone(), child.clone()]);
        let record = heap.datum(datum).unwrap();
        assert_eq!(record.type_path(), &runtime_type);
        assert_eq!(parent.type_path().as_str(), "/datum");
        assert_eq!(child.type_path().as_str(), "/datum/child");
        assert!(
            record
                .field(&field("name"))
                .unwrap()
                .semantic_eq(&text("child"))
        );
        let names: Vec<&str> = record.fields().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["name", "health", "speed"]);
    }

    #[test]
    fn datum_aliases_share_fields_and_stale_aliases_never_resolve() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/example").unwrap());
        let alias = datum;
        heap.set_datum_field(alias, field("value"), Value::number(8.0))
            .unwrap();
        assert!(
            heap.datum_field(datum, &field("value"))
                .unwrap()
                .semantic_eq(&Value::number(8.0))
        );

        heap.destroy_datum(datum).unwrap();
        let replacement = heap.allocate_datum(TypePath::parse("/datum/replacement").unwrap());
        assert_eq!(alias.index(), replacement.index());
        assert_ne!(alias.generation(), replacement.generation());
        assert_eq!(
            heap.set_datum_field(alias, field("value"), Value::Null),
            Err(ValueError::StaleDatum(alias))
        );
    }

    #[test]
    fn default_values_clone_handles_and_heap_iteration_is_snapshot_stable() {
        let mut heap = ValueHeap::new();
        let shared_list = heap.allocate_list();
        let mut defaults = DatumDefaults::new(TypePath::parse("/datum").unwrap());
        defaults.set(field("contents"), Value::List(shared_list));
        let first = heap.allocate_datum_with_defaults(
            TypePath::parse("/datum/first").unwrap(),
            std::slice::from_ref(&defaults),
        );
        let second = heap.allocate_datum_with_defaults(
            TypePath::parse("/datum/second").unwrap(),
            std::slice::from_ref(&defaults),
        );
        assert!(
            heap.datum_field(first, &field("contents"))
                .unwrap()
                .semantic_eq(heap.datum_field(second, &field("contents")).unwrap())
        );

        heap.destroy_datum(first).unwrap();
        let replacement = heap.allocate_datum(TypePath::parse("/datum/replacement").unwrap());
        let snapshot: Vec<(DatumId, &str)> = heap
            .datums()
            .map(|(id, datum)| (id, datum.type_path().as_str()))
            .collect();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[0], (replacement, "/datum/replacement"));
        assert_eq!(snapshot[1], (second, "/datum/second"));
    }

    #[test]
    fn equality_uses_value_content_and_logical_handles() {
        assert!(Value::number(0.0).semantic_eq(&Value::number(-0.0)));
        assert!(!Value::number(f32::NAN).semantic_eq(&Value::number(f32::NAN)));
        assert!(text("same").semantic_eq(&text("same")));
        assert!(!Value::Null.semantic_eq(&Value::number(0.0)));

        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        let alias = Value::List(list);
        assert!(alias.semantic_eq(&Value::List(list)));
    }

    #[test]
    fn positional_storage_is_one_based_and_shifts_after_removal() {
        let mut list = DmList::default();
        assert_eq!(list.add(text("first")), 1);
        assert_eq!(list.add(text("second")), 2);
        assert!(list.get(1).unwrap().semantic_eq(&text("first")));
        assert!(matches!(list.get(0), Err(ValueError::IndexZero)));
        assert!(matches!(
            list.get(3),
            Err(ValueError::IndexOutOfBounds { index: 3, len: 2 })
        ));

        let old = list.set(2, text("replacement")).unwrap();
        assert!(old.semantic_eq(&text("second")));
        assert!(list.remove(1).unwrap().semantic_eq(&text("first")));
        assert!(list.get(1).unwrap().semantic_eq(&text("replacement")));
    }

    #[test]
    fn removal_uses_semantic_equality_and_removes_only_first_match() {
        let mut list = DmList::default();
        list.add(Value::number(-0.0));
        list.add(Value::number(0.0));

        assert!(list.remove_first(&Value::number(0.0)).is_some());
        assert_eq!(list.len(), 1);
        assert!(list.get(1).unwrap().semantic_eq(&Value::number(0.0)));
    }

    #[test]
    fn assigning_an_existing_item_as_a_key_preserves_length_and_order() {
        let mut list = DmList::default();
        list.add(text("key"));
        list.add(text("other"));
        assert_eq!(list.set_key(text("key"), Value::number(7.0)), None);
        assert_eq!(list.len(), 2);
        assert!(list.get(1).unwrap().semantic_eq(&text("key")));
        assert!(
            list.get_key(&text("key"))
                .unwrap()
                .semantic_eq(&Value::number(7.0))
        );
    }

    #[test]
    fn large_associative_lists_keep_indexed_lookup_and_byond_order_in_sync() {
        assert_eq!(std::mem::size_of::<ListOrder>(), 4);
        assert_eq!(std::mem::size_of::<(u64, u32)>(), 16);
        let mut list = DmList::default();
        for index in 0..16 {
            list.set_key(
                Value::text(format!("signal-{index}")),
                Value::number(index as f32),
            );
        }
        assert_eq!(list.len(), 16);
        assert!(
            list.get_key(&text("signal-12"))
                .unwrap()
                .semantic_eq(&Value::number(12.0))
        );
        assert!(
            list.set_key(text("signal-12"), Value::number(99.0))
                .unwrap()
                .semantic_eq(&Value::number(12.0))
        );
        assert!(
            list.remove_key(&text("signal-3"))
                .unwrap()
                .semantic_eq(&Value::number(3.0))
        );
        assert!(
            list.get_key(&text("signal-12"))
                .unwrap()
                .semantic_eq(&Value::number(99.0))
        );
        assert_eq!(list.get(4).unwrap(), &text("signal-4"));

        let mut numeric = DmList::default();
        for index in 1..8 {
            numeric.set_key(Value::number(index as f32), Value::number(index as f32));
        }
        numeric.set_key(Value::number(-0.0), text("zero"));
        assert!(
            numeric
                .get_key(&Value::number(0.0))
                .unwrap()
                .semantic_eq(&text("zero"))
        );
    }

    #[test]
    fn compact_associative_index_collision_marker_preserves_exact_keys() {
        let mut list = DmList::default();
        for index in 0..16 {
            list.set_key(
                Value::text(format!("key-{index}")),
                Value::number(index as f32),
            );
        }

        let key = text("key-12");
        let hash = semantic_hash(&key).unwrap();
        list.associative_index
            .as_mut()
            .unwrap()
            .insert(hash, ASSOCIATIVE_INDEX_COLLISION);
        assert_eq!(list.get_key(&key), Ok(&Value::number(12.0)));
        assert!(list.contains(&key));
        assert_eq!(
            list.set_key(key.clone(), Value::number(99.0)),
            Some(Value::number(12.0)),
        );
        assert_eq!(list.get_key(&key), Ok(&Value::number(99.0)));

        let missing = text("missing-collision");
        list.associative_index.as_mut().unwrap().insert(
            semantic_hash(&missing).unwrap(),
            ASSOCIATIVE_INDEX_COLLISION,
        );
        assert_eq!(list.get_key(&missing), Err(ValueError::MissingKey));
    }

    #[test]
    fn empty_lists_keep_only_pointer_sized_lazy_storage() {
        assert_eq!(
            std::mem::size_of::<DmList>(),
            std::mem::size_of::<Option<Box<DmListStorage>>>()
        );
        let mut list = DmList::default();
        assert!(list.storage.is_none());
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(
            list.storage.is_none(),
            "read-only access stays allocation-free"
        );

        list.add(Value::number(7.0));
        assert!(list.storage.is_some());
        assert!(list.get(1).unwrap().semantic_eq(&Value::number(7.0)));
    }

    #[test]
    fn indexed_contains_preserves_mixed_list_semantics_and_avoids_key_scans() {
        use std::hint::black_box;
        use std::time::Instant;

        let mut list = DmList::default();
        list.add(text("ordinary"));
        for index in 0..4_096 {
            list.set_key(
                Value::text(format!("signal-{index}")),
                Value::number(index as f32),
            );
        }
        assert!(list.associative_index.is_some());
        assert!(list.contains(&text("ordinary")));
        assert!(list.contains(&text("signal-4095")));
        assert!(!list.contains(&text("missing")));

        let needle = text("signal-4095");
        let iterations = 10_000;
        let indexed_started = Instant::now();
        for _ in 0..iterations {
            black_box(list.contains(black_box(&needle)));
        }
        let indexed = indexed_started.elapsed();

        let linear_started = Instant::now();
        for _ in 0..iterations {
            black_box(
                list.positions()
                    .any(|(_, candidate)| candidate.semantic_eq(black_box(&needle))),
            );
        }
        let linear = linear_started.elapsed();
        eprintln!(
            "list-contains iterations={iterations} indexed={indexed:?} linear={linear:?} speedup={:.2}x",
            linear.as_secs_f64() / indexed.as_secs_f64()
        );
    }

    #[test]
    fn range_insert_swap_and_resize_preserve_order_and_associations() {
        let mut list = DmList::default();
        list.add(text("a"));
        list.set_key(text("key"), Value::number(9.0));
        list.add(text("b"));

        let copy = list.copy_range(2, 4).unwrap();
        assert_eq!(copy.len(), 2);
        assert!(copy.get(1).unwrap().semantic_eq(&text("key")));
        assert!(
            copy.get_key(&text("key"))
                .unwrap()
                .semantic_eq(&Value::number(9.0))
        );

        list.insert(2, text("x")).unwrap();
        assert_eq!(list.find_position(&text("b"), 1, list.len() + 1), Ok(4));
        list.swap(1, 4).unwrap();
        assert!(list.get(1).unwrap().semantic_eq(&text("b")));
        assert!(list.get(4).unwrap().semantic_eq(&text("a")));
        assert!(list.remove_last(&text("x")).is_some());
        list.resize(5).unwrap();
        assert_eq!(list.len(), 5);
        assert!(list.get(5).unwrap().semantic_eq(&Value::Null));
        list.resize(2).unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn cut_range_removes_mixed_entries_and_preserves_all_remaining_mappings() {
        let mut list = DmList::default();
        list.add(text("p0"));
        list.set_key(text("k0"), text("v0"));
        list.add(text("p1"));
        list.set_key(text("k1"), text("v1"));
        list.add(text("p2"));
        list.set_key(text("k2"), text("v2"));
        list.add(text("p3"));
        list.set_key(text("k3"), text("v3"));
        assert_eq!(list.cut_range(3, 7), Ok(4));
        assert_eq!(list.len(), 4);
        assert_eq!(list.positional_len(), 2);
        assert_eq!(list.associative_len(), 2);
        for (index, expected) in ["p0", "k0", "p3", "k3"].iter().enumerate() {
            assert!(list.get(index + 1).unwrap().semantic_eq(&text(expected)));
        }
        assert!(list.get_key(&text("k0")).unwrap().semantic_eq(&text("v0")));
        assert!(list.get_key(&text("k3")).unwrap().semantic_eq(&text("v3")));
        assert!(matches!(
            list.get_key(&text("k1")),
            Err(ValueError::MissingKey)
        ));
        assert!(list.associative_index.is_none());

        assert_eq!(list.cut_range(1, 2), Ok(1));
        assert_eq!(list.cut_range(3, 4), Ok(1));
        assert!(list.get(1).unwrap().semantic_eq(&text("k0")));
        assert!(list.get(2).unwrap().semantic_eq(&text("p3")));
        assert!(list.get_key(&text("k0")).unwrap().semantic_eq(&text("v0")));
    }

    #[test]
    fn whole_list_cut_releases_lazy_storage_and_empty_ranges_do_not_mutate() {
        let mut list = DmList::default();
        for index in 0..32 {
            list.add(Value::number(index as f32));
            list.set_key(
                Value::text(format!("key-{index}")),
                Value::number((index + 100) as f32),
            );
        }
        assert_eq!(list.cut_range(10, 10), Ok(0));
        assert_eq!(list.len(), 64);
        assert_eq!(list.cut_range(1, 65), Ok(64));
        assert!(list.is_empty());
        assert!(list.storage.is_none());
        assert_eq!(list.positional_len(), 0);
        assert_eq!(list.associative_len(), 0);

        list.add(text("reused"));
        assert!(list.get(1).unwrap().semantic_eq(&text("reused")));
    }

    #[test]
    fn indexed_repeated_remove_last_preserves_duplicates_order_and_mutation_semantics() {
        let mut list = DmList::default();
        for index in 0..128 {
            list.add(Value::number(index as f32));
        }
        list.add(Value::number(7.0));

        assert_eq!(
            list.remove_last(&Value::number(7.0)).unwrap().as_number(),
            Some(7.0)
        );
        assert!(list.positional_remove_index.is_some());
        assert_eq!(list.len(), 128);
        assert_eq!(list.get(8).unwrap().as_number(), Some(7.0));

        for index in (0..128).step_by(2) {
            assert_eq!(
                list.remove_last(&Value::number(index as f32))
                    .unwrap()
                    .as_number(),
                Some(index as f32)
            );
        }
        assert_eq!(list.len(), 64);
        for (position, expected) in (1..128).step_by(2).enumerate() {
            assert_eq!(
                list.get(position + 1).unwrap().as_number(),
                Some(expected as f32)
            );
        }
        assert!(list.remove_last(&Value::number(200.0)).is_none());

        list.add(text("tail"));
        assert!(list.positional_remove_index.is_none());
        assert!(
            list.remove_last(&text("tail"))
                .unwrap()
                .semantic_eq(&text("tail"))
        );
        list.swap(1, 2).unwrap();
        assert_eq!(list.get(1).unwrap().as_number(), Some(3.0));
        assert_eq!(list.get(2).unwrap().as_number(), Some(1.0));
    }

    #[test]
    fn associative_updates_preserve_deterministic_insertion_order() {
        let mut list = DmList::default();
        assert!(list.set_key(text("first"), Value::number(1.0)).is_none());
        assert!(list.set_key(text("second"), Value::number(2.0)).is_none());
        let old = list.set_key(text("first"), Value::number(3.0)).unwrap();
        assert!(old.semantic_eq(&Value::number(1.0)));
        assert_eq!(list.associative_len(), 2);
        assert!(
            list.get_key(&text("first"))
                .unwrap()
                .semantic_eq(&Value::number(3.0))
        );
        assert!(matches!(
            list.get_key(&text("absent")),
            Err(ValueError::MissingKey)
        ));

        let keys: Vec<&str> = list
            .associations()
            .map(|(key, _)| match key {
                Value::Text(value) => value.as_ref(),
                _ => panic!("test keys are text"),
            })
            .collect();
        assert_eq!(keys, ["first", "second"]);
    }

    #[test]
    fn positional_values_and_associative_keys_share_one_iteration_order() {
        let mut list = DmList::default();
        list.add(Value::number(1.0));
        list.set_key(text("key"), Value::number(10.0));
        list.add(Value::number(2.0));

        assert_eq!(list.len(), 3);
        assert_eq!(list.positional_len(), 2);
        assert!(list.get(1).unwrap().semantic_eq(&Value::number(1.0)));
        assert!(list.get(2).unwrap().semantic_eq(&text("key")));
        assert!(list.get(3).unwrap().semantic_eq(&Value::number(2.0)));
        assert_eq!(
            list.set(2, Value::number(3.0)),
            Err(ValueError::AssociativeIndexAssignment { index: 2 })
        );
    }

    #[test]
    fn aliases_share_identity_while_copies_are_shallow_and_distinct() {
        let mut heap = ValueHeap::new();
        let child = heap.allocate_list();
        let original = heap.allocate_list();
        heap.list_mut(original).unwrap().add(Value::List(child));

        let alias = original;
        heap.list_mut(alias).unwrap().add(text("shared"));
        assert_eq!(heap.list(original).unwrap().len(), 2);

        let copy = heap.copy_list(original).unwrap();
        assert_ne!(copy, original);
        assert!(Arc::ptr_eq(
            heap.list(original).unwrap().storage.as_ref().unwrap(),
            heap.list(copy).unwrap().storage.as_ref().unwrap(),
        ));
        assert!(
            heap.list(copy)
                .unwrap()
                .get(1)
                .unwrap()
                .semantic_eq(&Value::List(child))
        );
        heap.list_mut(copy).unwrap().add(text("copy only"));
        assert!(!Arc::ptr_eq(
            heap.list(original).unwrap().storage.as_ref().unwrap(),
            heap.list(copy).unwrap().storage.as_ref().unwrap(),
        ));
        assert_eq!(heap.list(original).unwrap().len(), 2);
        assert_eq!(heap.list(copy).unwrap().len(), 3);
    }

    #[test]
    fn generations_reject_stale_handles_after_slot_reuse() {
        let mut heap = ValueHeap::new();
        let old_list = heap.allocate_list();
        heap.destroy_list(old_list).unwrap();
        let new_list = heap.allocate_list();
        assert_eq!(old_list.index(), new_list.index());
        assert_ne!(old_list.generation(), new_list.generation());
        assert!(matches!(
            heap.list(old_list),
            Err(ValueError::StaleList(id)) if id == old_list
        ));

        let path = TypePath::parse("/datum/test").unwrap();
        let old_datum = heap.allocate_datum(path.clone());
        assert_eq!(heap.datum(old_datum).unwrap().type_path(), &path);
        heap.destroy_datum(old_datum).unwrap();
        let new_datum = heap.allocate_datum(TypePath::parse("/datum/new").unwrap());
        assert_eq!(old_datum.index(), new_datum.index());
        assert_ne!(old_datum.generation(), new_datum.generation());
        assert_eq!(
            heap.datum(old_datum),
            Err(ValueError::StaleDatum(old_datum))
        );
    }

    #[test]
    fn arena_growth_is_chunked_and_handles_cross_chunk_boundaries() {
        let mut arena = Arena::default();
        let mut handles = Vec::with_capacity(ARENA_CHUNK_SLOTS + 1);
        for value in 0..=ARENA_CHUNK_SLOTS {
            handles.push(arena.insert(value));
        }

        assert_eq!(arena.chunks.len(), 2);
        assert_eq!(arena.chunks[0].len(), ARENA_CHUNK_SLOTS);
        assert_eq!(arena.chunks[1].len(), 1);
        let (last_index, last_generation) = handles[ARENA_CHUNK_SLOTS];
        assert_eq!(
            arena.get(last_index, last_generation),
            Some(&ARENA_CHUNK_SLOTS)
        );

        let (old_index, old_generation) = handles[ARENA_CHUNK_SLOTS - 1];
        assert_eq!(
            arena.remove(old_index, old_generation),
            Some(ARENA_CHUNK_SLOTS - 1)
        );
        let (reused_index, reused_generation) = arena.insert(99_999);
        assert_eq!(reused_index, old_index);
        assert_ne!(reused_generation, old_generation);
        assert_eq!(arena.get(reused_index, reused_generation), Some(&99_999));
    }

    #[test]
    fn truth_rules_validate_reference_liveness() {
        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        let datum = heap.allocate_datum(TypePath::parse("/datum/test").unwrap());

        for false_value in [Value::Null, Value::number(0.0), text("")] {
            assert!(!heap.truthy(&false_value).unwrap());
        }
        for true_value in [
            Value::number(-1.0),
            text("x"),
            Value::TypePath(TypePath::parse("/obj").unwrap()),
            Value::List(list),
            Value::Datum(datum),
        ] {
            assert!(heap.truthy(&true_value).unwrap());
        }

        heap.destroy_list(list).unwrap();
        heap.destroy_datum(datum).unwrap();
        assert!(!heap.truthy(&Value::List(list)).unwrap());
        assert!(!heap.truthy(&Value::Datum(datum)).unwrap());
    }

    #[test]
    fn iteration_reports_positions_in_ascending_order() {
        let mut list = DmList::default();
        list.add(text("one"));
        list.add(text("two"));
        list.add(text("three"));
        let positions: Vec<usize> = list.positions().map(|(index, _)| index).collect();
        assert_eq!(positions, [1, 2, 3]);
    }

    #[test]
    fn repeated_prefix_cuts_preserve_logical_indexing_and_copy_ranges() {
        let mut list = DmList::default();
        for value in 0..10_000 {
            list.add(Value::number(value as f32));
        }
        for expected in 0..9_900 {
            assert_eq!(list.get(1).unwrap(), &Value::number(expected as f32));
            assert_eq!(list.cut_range(1, 2), Ok(1));
        }
        assert_eq!(list.len(), 100);
        assert_eq!(list.positional_len(), 100);
        assert_eq!(list.get(1).unwrap(), &Value::number(9_900.0));
        assert_eq!(list.get(100).unwrap(), &Value::number(9_999.0));
        let copy = list.copy_range(2, 5).unwrap();
        assert_eq!(copy.len(), 3);
        assert_eq!(copy.get(1).unwrap(), &Value::number(9_901.0));
        assert_eq!(copy.get(3).unwrap(), &Value::number(9_903.0));
    }

    #[test]
    fn mutations_after_lazy_prefix_cut_materialize_without_changing_semantics() {
        let mut list = DmList::default();
        for value in 1..=8 {
            list.add(Value::number(value as f32));
        }
        list.cut_range(1, 4).unwrap();
        list.set(1, Value::number(50.0)).unwrap();
        list.insert(2, Value::number(60.0)).unwrap();
        list.swap(1, 3).unwrap();
        list.set_key(text("key"), text("value"));
        assert_eq!(list.len(), 7);
        assert_eq!(list.get(1).unwrap(), &Value::number(5.0));
        assert_eq!(list.get(2).unwrap(), &Value::number(60.0));
        assert_eq!(list.get(3).unwrap(), &Value::number(50.0));
        assert_eq!(list.get_key(&text("key")).unwrap(), &text("value"));
        assert_eq!(list.remove(1).unwrap(), Value::number(5.0));
        assert_eq!(list.get(1).unwrap(), &Value::number(60.0));
    }

    #[test]
    fn cloned_list_storage_detaches_before_advancing_lazy_prefix() {
        let mut original = DmList::default();
        for value in 1..=4 {
            original.add(Value::number(value as f32));
        }
        let snapshot = original.clone();
        original.cut_range(1, 3).unwrap();
        assert_eq!(original.len(), 2);
        assert_eq!(original.get(1).unwrap(), &Value::number(3.0));
        assert_eq!(snapshot.len(), 4);
        assert_eq!(snapshot.get(1).unwrap(), &Value::number(1.0));
    }

    #[test]
    fn gc_compaction_releases_removed_arc_payloads() {
        let mut heap = ValueHeap::new();
        let queue = heap.allocate_list();
        let payload: Arc<str> = Arc::from("retained payload");
        for _ in 0..2_048 {
            heap.list_mut(queue)
                .unwrap()
                .add(Value::Text(Arc::clone(&payload)));
        }
        heap.list_mut(queue).unwrap().cut_range(1, 1_025).unwrap();
        assert_eq!(
            Arc::strong_count(&payload),
            2_049,
            "a lazy cut must not add hot-path destruction work"
        );

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[queue]);
        assert_eq!(stats.list_storage.compacted_lists, 1);
        assert_eq!(stats.list_storage.compacted_prefix_entries, 1_024);
        assert_eq!(stats.list_storage.prefix_retained, 0);
        assert_eq!(
            Arc::strong_count(&payload),
            1_025,
            "GC compaction must drop Values retained only by the dead prefix"
        );
    }

    #[test]
    fn gc_prefix_compaction_preserves_cow_source_order_and_associations() {
        let mut heap = ValueHeap::new();
        let source = heap.allocate_list();
        for value in 1..=4_096 {
            heap.list_mut(source)
                .unwrap()
                .add(Value::number(value as f32));
        }
        let queue = heap.copy_list(source).unwrap();
        heap.list_mut(queue).unwrap().cut_range(1, 2_049).unwrap();

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[source, queue]);
        assert_eq!(stats.list_storage.compacted_lists, 1);
        assert_eq!(heap.list(source).unwrap().len(), 4_096);
        assert_eq!(heap.list(source).unwrap().get(1), Ok(&Value::number(1.0)));

        let queue = heap.list_mut(queue).unwrap();
        queue.set_key(text("key"), text("associated"));
        queue.add(text("tail"));
        assert_eq!(queue.len(), 2_050);
        assert_eq!(queue.get(1), Ok(&Value::number(2_049.0)));
        assert_eq!(queue.get(2_048), Ok(&Value::number(4_096.0)));
        assert_eq!(queue.get(2_049), Ok(&text("key")));
        assert_eq!(queue.get(2_050), Ok(&text("tail")));
        assert_eq!(queue.get_key(&text("key")), Ok(&text("associated")));
    }

    #[test]
    fn gc_prefix_compaction_detaches_shared_lazy_storage() {
        let mut queue = DmList::default();
        for value in 1..=4_096 {
            queue.add(Value::number(value as f32));
        }
        queue.cut_range(1, 2_049).unwrap();
        let source = queue.clone();
        let shared_storage = Arc::as_ptr(source.storage.as_ref().unwrap());
        let source_capacity = source.positional.capacity();
        assert_eq!(Arc::as_ptr(queue.storage.as_ref().unwrap()), shared_storage);

        let mut stats = ListStorageStats::default();
        queue.compact_and_measure_for_gc(&mut stats);
        assert_eq!(stats.compacted_lists, 1);
        assert_eq!(queue.prefix_head, 0);
        assert_eq!(queue.len(), 2_048);
        assert_eq!(queue.get(1), Ok(&Value::number(2_049.0)));
        assert_ne!(Arc::as_ptr(queue.storage.as_ref().unwrap()), shared_storage);

        assert_eq!(source.prefix_head, 2_048);
        assert_eq!(source.len(), 2_048);
        assert_eq!(source.positional.capacity(), source_capacity);
        assert_eq!(
            Arc::as_ptr(source.storage.as_ref().unwrap()),
            shared_storage
        );
        assert_eq!(source.get(1), Ok(&Value::number(2_049.0)));
    }

    #[test]
    fn gc_does_not_churn_small_lazy_prefixes() {
        let mut heap = ValueHeap::new();
        let queue = heap.allocate_list();
        for value in 0..14_000 {
            heap.list_mut(queue)
                .unwrap()
                .add(Value::number(value as f32));
        }
        heap.list_mut(queue).unwrap().cut_range(1, 33).unwrap();
        let storage_before = Arc::as_ptr(heap.list(queue).unwrap().storage.as_ref().unwrap());
        let capacity_before = heap.list(queue).unwrap().positional.capacity();

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[queue]);
        let list = heap.list(queue).unwrap();
        assert_eq!(stats.list_storage.compacted_lists, 0);
        assert_eq!(stats.list_storage.prefix_retained, 32);
        assert_eq!(list.prefix_head, 32);
        assert_eq!(list.positional.capacity(), capacity_before);
        assert_eq!(Arc::as_ptr(list.storage.as_ref().unwrap()), storage_before);
    }

    #[test]
    fn gc_compacts_representative_93k_queue_capacity() {
        let mut heap = ValueHeap::new();
        let queue = heap.allocate_list();
        for value in 0..93_000 {
            heap.list_mut(queue)
                .unwrap()
                .add(Value::number(value as f32));
        }
        heap.list_mut(queue).unwrap().cut_range(1, 90_001).unwrap();
        let before_capacity_bytes = {
            let list = heap.list(queue).unwrap();
            list.positional
                .capacity()
                .saturating_mul(std::mem::size_of::<Value>())
                .saturating_add(
                    list.order
                        .capacity()
                        .saturating_mul(std::mem::size_of::<ListOrder>()),
                )
        };
        let started = std::time::Instant::now();
        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[queue]);
        let elapsed = started.elapsed();
        let list = heap.list(queue).unwrap();
        let after_capacity_bytes = list
            .positional
            .capacity()
            .saturating_mul(std::mem::size_of::<Value>())
            .saturating_add(
                list.order
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ListOrder>()),
            );
        eprintln!(
            "93k GC queue compaction: capacity_bytes={before_capacity_bytes}->{after_capacity_bytes} elapsed_us={}",
            elapsed.as_micros()
        );
        assert_eq!(list.len(), 3_000);
        assert_eq!(list.get(1), Ok(&Value::number(90_000.0)));
        assert_eq!(stats.list_storage.compacted_prefix_entries, 90_000);
        assert_eq!(stats.list_storage.shrunk_vectors, 2);
        assert!(after_capacity_bytes < before_capacity_bytes / 10);
    }

    #[test]
    fn gc_reclaims_boot_scale_tail_slack_with_growth_headroom() {
        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        for value in 0..93_000 {
            heap.list_mut(list)
                .unwrap()
                .add(Value::number(value as f32));
        }
        // Model a long-lived startup vector that peaked above its final live
        // length. This is the aggregate shape reported immediately before the
        // Boot203 Lighting OOM (roughly 1.5x capacity).
        {
            let list = heap.list_mut(list).unwrap();
            list.positional.truncate(87_000);
            list.order.truncate(87_000);
        }
        let before_capacity_bytes = {
            let list = heap.list(list).unwrap();
            list.positional
                .capacity()
                .saturating_mul(std::mem::size_of::<Value>())
                .saturating_add(
                    list.order
                        .capacity()
                        .saturating_mul(std::mem::size_of::<ListOrder>()),
                )
        };

        let started = std::time::Instant::now();
        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[list]);
        let elapsed = started.elapsed();
        let list = heap.list(list).unwrap();
        let after_capacity_bytes = list
            .positional
            .capacity()
            .saturating_mul(std::mem::size_of::<Value>())
            .saturating_add(
                list.order
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ListOrder>()),
            );
        eprintln!(
            "93k tail-slack GC: capacity_bytes={before_capacity_bytes}->{after_capacity_bytes} reclaimed={} elapsed_us={}",
            stats.list_storage.reclaimed_capacity_bytes,
            elapsed.as_micros(),
        );

        assert_eq!(stats.list_storage.shrunk_vectors, 2);
        assert_eq!(
            before_capacity_bytes.saturating_sub(after_capacity_bytes),
            stats.list_storage.reclaimed_capacity_bytes,
        );
        assert!(stats.list_storage.reclaimed_capacity_bytes >= 512 * 1_024);
        assert!(list.positional.capacity() > list.positional.len());
        assert!(list.order.capacity() > list.order.len());
        assert_eq!(list.get(1), Ok(&Value::number(0.0)));
        assert_eq!(list.get(87_000), Ok(&Value::number(86_999.0)));
    }

    #[test]
    fn parallel_gc_root_validation_preserves_source_order() {
        let roots = (0_u32..40_000).collect::<Vec<_>>();
        let validated = validate_gc_roots_parallel(&roots, |root| root % 3 == 0);
        let expected = roots
            .iter()
            .copied()
            .filter(|root| root % 3 == 0)
            .collect::<Vec<_>>();
        assert_eq!(validated, expected);
    }

    #[test]
    #[ignore = "local release multicore GC benchmark"]
    fn gc_parallel_root_validation_release_benchmark() {
        const ROOTS: usize = 1_000_000;
        const LIVE: usize = 100_000;
        let mut heap = ValueHeap::new();
        let live = (0..LIVE)
            .map(|_| heap.allocate_datum(TypePath::parse("/datum/gc_root").unwrap()))
            .collect::<Vec<_>>();
        let roots = (0..ROOTS)
            .map(|index| live[index % LIVE])
            .collect::<Vec<_>>();
        // Initialize the persistent pool outside the measurement.
        let _ = validate_gc_roots_parallel(&roots, |root| heap.datum(root).is_ok());
        let started = std::time::Instant::now();
        let sequential = roots
            .iter()
            .copied()
            .filter(|root| heap.datum(*root).is_ok())
            .collect::<Vec<_>>();
        let sequential_elapsed = started.elapsed();
        let started = std::time::Instant::now();
        let parallel = validate_gc_roots_parallel(&roots, |root| heap.datum(root).is_ok());
        let parallel_elapsed = started.elapsed();
        assert_eq!(parallel, sequential);
        eprintln!(
            "GC root validation roots={ROOTS} workers={} sequential_us={} parallel_us={} speedup={:.2}",
            std::thread::available_parallelism().map_or(1, usize::from),
            sequential_elapsed.as_micros(),
            parallel_elapsed.as_micros(),
            sequential_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "local release memory benchmark"]
    fn gc_millions_small_vector_release_benchmark() {
        const SAMPLE: usize = 100_000;
        let mut heap = ValueHeap::new();
        let mut datum_roots = Vec::with_capacity(SAMPLE);
        let mut positional_roots = Vec::with_capacity(SAMPLE);
        let mut associative_roots = Vec::with_capacity(SAMPLE);
        let datum_field = field("value");
        for _ in 0..SAMPLE {
            let positional = heap.allocate_list();
            heap.list_mut(positional).unwrap().add(Value::Null);
            positional_roots.push(positional);

            let associative = heap.allocate_list();
            heap.list_mut(associative)
                .unwrap()
                .set_key(Value::Null, Value::Null);
            associative_roots.push(associative);

            let datum = heap.allocate_datum(TypePath::parse("/datum/small_vector").unwrap());
            heap.set_datum_field(datum, datum_field.clone(), Value::Null)
                .unwrap();
            datum_roots.push(datum);
        }

        let started = std::time::Instant::now();
        let mut list_roots = positional_roots.clone();
        list_roots.extend_from_slice(&associative_roots);
        let stats = heap.collect_unreachable_values_from_ids_with_stats(&datum_roots, &list_roots);
        let elapsed = started.elapsed();
        let positional_bytes = positional_roots
            .iter()
            .map(|list| {
                let list = heap.list(*list).unwrap();
                (4usize.saturating_sub(list.positional.capacity()))
                    .saturating_mul(std::mem::size_of::<Value>())
            })
            .sum::<usize>();
        let associative_bytes = associative_roots
            .iter()
            .map(|list| {
                let list = heap.list(*list).unwrap();
                (4usize.saturating_sub(list.associative.capacity()))
                    .saturating_mul(std::mem::size_of::<(Value, Value)>())
            })
            .sum::<usize>();
        let datum_bytes = datum_roots
            .iter()
            .map(|datum| {
                let datum = heap.datum(*datum).unwrap();
                (4usize.saturating_sub(datum.fields.capacity()))
                    .saturating_mul(std::mem::size_of::<(FieldName, Value)>())
            })
            .sum::<usize>();
        eprintln!(
            "small-vector GC sample={SAMPLE} positional_bytes={positional_bytes} associative_bytes={associative_bytes} datum_bytes={datum_bytes} total_reported={} elapsed_ms={} extrapolated_boot_positional_mib={} extrapolated_boot_datum_mib={}",
            stats
                .list_storage
                .reclaimed_capacity_bytes
                .saturating_add(stats.datum_storage.reclaimed_capacity_bytes),
            elapsed.as_millis(),
            positional_bytes
                .saturating_mul(1_923_680)
                .saturating_div(SAMPLE)
                .saturating_div(1024 * 1024),
            datum_bytes
                .saturating_mul(1_522_931)
                .saturating_div(SAMPLE)
                .saturating_div(1024 * 1024),
        );
        assert_eq!(stats.list_storage.shrunk_vectors, SAMPLE * 2);
        assert_eq!(stats.datum_storage.shrunk_field_vectors, SAMPLE);
        assert_eq!(
            stats.list_storage.reclaimed_capacity_bytes,
            positional_bytes.saturating_add(associative_bytes),
        );
        assert_eq!(stats.datum_storage.reclaimed_capacity_bytes, datum_bytes);
    }

    #[test]
    #[ignore = "local release index-capacity benchmark"]
    fn gc_distributed_hash_index_release_benchmark() {
        const SAMPLE: usize = 10_000;
        const FIELD_LAYOUTS: usize = 100;
        const LOOKUP_ROUNDS: usize = 50;
        let field_names = (0..64)
            .map(|index| field(&format!("field_{index:02}")))
            .collect::<Vec<_>>();
        let list_keys = (0..64)
            .map(|index| text(&format!("key-{index:02}")))
            .collect::<Vec<_>>();
        let mut heap = ValueHeap::new();
        let mut datum_roots = Vec::with_capacity(SAMPLE);
        let mut list_roots = Vec::with_capacity(SAMPLE);
        let mut datum_capacity_before = 0usize;
        let mut list_capacity_before = 0usize;

        for sample_index in 0..SAMPLE {
            let datum = heap.allocate_datum(TypePath::parse("/datum/index_benchmark").unwrap());
            for (index, name) in field_names[..31].iter().enumerate() {
                heap.set_datum_field(datum, name.clone(), Value::number(index as f32))
                    .unwrap();
            }
            heap.set_datum_field(
                datum,
                field(&format!("layout_{}", sample_index % FIELD_LAYOUTS)),
                Value::number(31.0),
            )
            .unwrap();
            for (index, name) in field_names[32..].iter().enumerate() {
                heap.set_datum_field(datum, name.clone(), Value::number((index + 32) as f32))
                    .unwrap();
            }
            {
                let datum = heap.datum_mut(datum).unwrap();
                datum.fields.truncate(32);
                Arc::make_mut(datum.field_index.as_mut().unwrap())
                    .retain(|_, position| *position < 32);
                datum_capacity_before = datum_capacity_before
                    .saturating_add(datum.field_index.as_ref().unwrap().capacity());
            }
            datum_roots.push(datum);

            let list = heap.allocate_list();
            for (index, key) in list_keys.iter().enumerate() {
                heap.list_mut(list)
                    .unwrap()
                    .set_key(key.clone(), Value::number(index as f32));
            }
            {
                let list = heap.list_mut(list).unwrap();
                list.associative.truncate(32);
                list.order.truncate(32);
                list.associative_index
                    .as_mut()
                    .unwrap()
                    .retain(|_, position| *position < 32);
                list_capacity_before = list_capacity_before
                    .saturating_add(list.associative_index.as_ref().unwrap().capacity());
            }
            list_roots.push(list);
        }

        let measure_lookups = |heap: &ValueHeap| {
            let started = std::time::Instant::now();
            let mut hits = 0usize;
            for round in 0..LOOKUP_ROUNDS {
                let field_name = &field_names[round % 31];
                let list_key = &list_keys[round % 32];
                for datum in &datum_roots {
                    hits = hits.saturating_add(usize::from(
                        heap.datum(*datum).unwrap().field(field_name).is_ok(),
                    ));
                }
                for list in &list_roots {
                    hits = hits.saturating_add(usize::from(
                        heap.list(*list).unwrap().get_key(list_key).is_ok(),
                    ));
                }
            }
            std::hint::black_box(hits);
            started.elapsed()
        };
        let lookup_before = measure_lookups(&heap);
        let started = std::time::Instant::now();
        let stats = heap.collect_unreachable_values_from_ids_with_stats(&datum_roots, &list_roots);
        let gc_elapsed = started.elapsed();
        let lookup_after = measure_lookups(&heap);
        let repeat_started = std::time::Instant::now();
        let repeat_stats =
            heap.collect_unreachable_values_from_ids_with_stats(&datum_roots, &list_roots);
        let repeat_gc_elapsed = repeat_started.elapsed();
        let datum_capacity_after = datum_roots
            .iter()
            .map(|datum| {
                heap.datum(*datum)
                    .unwrap()
                    .field_index
                    .as_ref()
                    .unwrap()
                    .capacity()
            })
            .sum::<usize>();
        let list_capacity_after = list_roots
            .iter()
            .map(|list| {
                heap.list(*list)
                    .unwrap()
                    .associative_index
                    .as_ref()
                    .unwrap()
                    .capacity()
            })
            .sum::<usize>();

        eprintln!(
            "distributed-index GC sample={SAMPLE} field_layouts={FIELD_LAYOUTS} datum_slots={datum_capacity_before}->{datum_capacity_after} datum_shrink_bytes_reclaimed={} datum_indexes_deduplicated={} datum_dedupe_bytes_reclaimed={} datum_physical_indexes={} first_fingerprints={} first_pointer_hits={} first_exact_compares={} repeat_fingerprints={} repeat_pointer_hits={} repeat_exact_compares={} list_slots={list_capacity_before}->{list_capacity_after} list_entry_bytes_reclaimed={} first_gc_ms={} repeat_gc_ms={} lookups={} lookup_before_ms={} lookup_after_ms={}",
            stats.datum_storage.reclaimed_field_index_bytes,
            stats.datum_storage.deduplicated_field_indexes,
            stats.datum_storage.deduplicated_field_index_bytes,
            stats.datum_storage.physical_field_indexes,
            stats.datum_storage.field_index_fingerprints_computed,
            stats.datum_storage.field_index_pointer_cache_hits,
            stats.datum_storage.field_index_exact_layout_comparisons,
            repeat_stats.datum_storage.field_index_fingerprints_computed,
            repeat_stats.datum_storage.field_index_pointer_cache_hits,
            repeat_stats
                .datum_storage
                .field_index_exact_layout_comparisons,
            stats.list_storage.reclaimed_associative_index_bytes,
            gc_elapsed.as_millis(),
            repeat_gc_elapsed.as_millis(),
            SAMPLE * LOOKUP_ROUNDS * 2,
            lookup_before.as_millis(),
            lookup_after.as_millis(),
        );
        assert_eq!(
            datum_capacity_before.saturating_sub(datum_capacity_after),
            stats.datum_storage.reclaimed_field_index_capacity,
        );
        assert_eq!(
            list_capacity_before.saturating_sub(list_capacity_after),
            stats.list_storage.reclaimed_associative_index_capacity,
        );
        assert_eq!(stats.datum_storage.shrunk_field_indexes, SAMPLE);
        assert_eq!(
            stats.datum_storage.deduplicated_field_indexes,
            SAMPLE - FIELD_LAYOUTS,
        );
        assert_eq!(stats.datum_storage.physical_field_indexes, FIELD_LAYOUTS,);
        assert_eq!(
            stats.datum_storage.field_index_fingerprints_computed,
            SAMPLE,
        );
        assert_eq!(stats.datum_storage.field_index_pointer_cache_hits, 0);
        assert_eq!(
            stats.datum_storage.field_index_exact_layout_comparisons,
            SAMPLE - FIELD_LAYOUTS,
        );
        assert_eq!(repeat_stats.datum_storage.deduplicated_field_indexes, 0);
        assert_eq!(
            repeat_stats.datum_storage.physical_field_indexes,
            FIELD_LAYOUTS,
        );
        assert_eq!(
            repeat_stats.datum_storage.field_index_fingerprints_computed,
            FIELD_LAYOUTS,
        );
        assert_eq!(
            repeat_stats.datum_storage.field_index_pointer_cache_hits,
            SAMPLE - FIELD_LAYOUTS,
        );
        assert_eq!(
            repeat_stats
                .datum_storage
                .field_index_exact_layout_comparisons,
            0,
        );
        assert_eq!(stats.list_storage.shrunk_associative_indexes, SAMPLE);
        assert!(datum_capacity_after >= SAMPLE * 40);
        assert!(list_capacity_after >= SAMPLE * 40);
    }

    #[test]
    fn gc_capacity_reclamation_preserves_associations_and_hot_index() {
        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        for index in 0..2_048 {
            heap.list_mut(list).unwrap().set_key(
                text(&format!("key-{index:04}")),
                Value::number(index as f32),
            );
        }
        {
            let list = heap.list_mut(list).unwrap();
            list.associative.truncate(1_024);
            list.order.truncate(1_024);
            list.associative_index
                .as_mut()
                .unwrap()
                .retain(|_, position| *position < 1_024);
        }

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[list]);
        let list = heap.list(list).unwrap();
        assert!(stats.list_storage.shrunk_vectors >= 1);
        assert_eq!(stats.list_storage.associative_indexes, 1);
        assert_eq!(stats.list_storage.shrunk_associative_indexes, 1);
        assert!(stats.list_storage.reclaimed_associative_index_capacity > 0);
        assert!(
            list.associative_index.is_some(),
            "the hot index stays built"
        );
        assert_eq!(list.len(), 1_024);
        assert_eq!(list.get_key(&text("key-0000")), Ok(&Value::number(0.0)),);
        assert_eq!(list.get_key(&text("key-1023")), Ok(&Value::number(1_023.0)),);
        assert_eq!(list.get_key(&text("key-1024")), Err(ValueError::MissingKey),);
    }

    #[test]
    fn gc_reclaims_distributed_association_index_slack_below_old_floor() {
        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        for index in 0..64 {
            heap.list_mut(list).unwrap().set_key(
                text(&format!("key-{index:02}")),
                Value::number(index as f32),
            );
        }
        {
            let list = heap.list_mut(list).unwrap();
            list.associative.truncate(52);
            list.order.truncate(52);
            list.associative_index
                .as_mut()
                .unwrap()
                .retain(|_, position| *position < 52);
        }
        let before = heap
            .list(list)
            .unwrap()
            .associative_index
            .as_ref()
            .unwrap()
            .capacity();
        assert!(before.saturating_sub(52) < 128);

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[list]);
        let list_record = heap.list(list).unwrap();
        let after = list_record.associative_index.as_ref().unwrap().capacity();
        let reclaimed = before.saturating_sub(after);
        assert!(reclaimed >= 8);
        assert_eq!(stats.list_storage.shrunk_associative_indexes, 1);
        assert_eq!(
            stats.list_storage.reclaimed_associative_index_capacity,
            reclaimed,
        );
        assert_eq!(
            stats.list_storage.reclaimed_associative_index_bytes,
            reclaimed.saturating_mul(std::mem::size_of::<(u64, u32)>()),
        );
        assert!(after >= 56, "the compacted index retains 6.25% headroom");
        assert_eq!(
            list_record.get_key(&text("key-51")),
            Ok(&Value::number(51.0)),
        );

        let retained_capacity = after;
        for index in 64..68 {
            heap.list_mut(list).unwrap().set_key(
                text(&format!("key-{index:02}")),
                Value::number(index as f32),
            );
        }
        assert_eq!(
            heap.list(list)
                .unwrap()
                .associative_index
                .as_ref()
                .unwrap()
                .capacity(),
            retained_capacity,
            "6.25% immediate growth must not reallocate the index",
        );
    }

    #[test]
    fn gc_drops_unique_rebuildable_positional_remove_index() {
        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        for index in 0..256 {
            heap.list_mut(list)
                .unwrap()
                .add(Value::number(index as f32));
        }
        assert_eq!(
            heap.list_mut(list)
                .unwrap()
                .remove_last(&Value::number(255.0)),
            Some(Value::number(255.0)),
        );
        assert!(heap.list(list).unwrap().positional_remove_index.is_some());

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[list]);
        assert_eq!(stats.list_storage.positional_remove_indexes, 1);
        assert_eq!(stats.list_storage.dropped_positional_remove_indexes, 1);
        assert!(stats.list_storage.positional_remove_key_len > 0);
        assert!(stats.list_storage.positional_remove_position_len > 0);
        assert!(stats.list_storage.positional_remove_removed_len > 0);
        assert!(heap.list(list).unwrap().positional_remove_index.is_none());

        assert_eq!(
            heap.list_mut(list)
                .unwrap()
                .remove_last(&Value::number(254.0)),
            Some(Value::number(254.0)),
            "the transient index must rebuild without changing semantics",
        );
        assert!(heap.list(list).unwrap().positional_remove_index.is_some());
    }

    #[test]
    fn gc_does_not_detach_shared_cow_storage_to_reclaim_capacity_or_indexes() {
        let mut heap = ValueHeap::new();
        let source = heap.allocate_list();
        for index in 0..8_192 {
            heap.list_mut(source)
                .unwrap()
                .add(Value::number(index as f32));
        }
        {
            let source = heap.list_mut(source).unwrap();
            source.positional.truncate(4_096);
            source.order.truncate(4_096);
            assert_eq!(
                source.remove_last(&Value::number(4_095.0)),
                Some(Value::number(4_095.0)),
            );
        }
        let copy = heap.copy_list(source).unwrap();
        let pointer = Arc::as_ptr(heap.list(source).unwrap().storage.as_ref().unwrap());
        let positional_capacity = heap.list(source).unwrap().positional.capacity();

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[source, copy]);
        let source_list = heap.list(source).unwrap();
        let copy_list = heap.list(copy).unwrap();
        assert_eq!(stats.list_storage.shrunk_vectors, 0);
        assert_eq!(stats.list_storage.reclaimed_capacity_bytes, 0);
        assert!(stats.list_storage.shared_shrink_candidates >= 2);
        assert!(stats.list_storage.shared_derived_index_candidates >= 2);
        assert_eq!(Arc::as_ptr(source_list.storage.as_ref().unwrap()), pointer);
        assert_eq!(Arc::as_ptr(copy_list.storage.as_ref().unwrap()), pointer);
        assert_eq!(source_list.positional.capacity(), positional_capacity);
        assert!(source_list.positional_remove_index.is_some());
        assert!(copy_list.positional_remove_index.is_some());
        assert_eq!(source_list.get(1), copy_list.get(1));
        assert_eq!(
            source_list.get(source_list.len()),
            copy_list.get(copy_list.len())
        );
    }

    #[test]
    fn gc_does_not_churn_sub_page_capacity_slack() {
        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        for index in 0..4 {
            heap.list_mut(list)
                .unwrap()
                .add(Value::number(index as f32));
        }
        {
            let list = heap.list_mut(list).unwrap();
            list.positional.truncate(2);
            list.order.truncate(2);
        }
        let pointer = Arc::as_ptr(heap.list(list).unwrap().storage.as_ref().unwrap());
        let capacity = heap.list(list).unwrap().positional.capacity();

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[], &[list]);
        let list = heap.list(list).unwrap();
        assert_eq!(stats.list_storage.shrunk_vectors, 0);
        assert_eq!(Arc::as_ptr(list.storage.as_ref().unwrap()), pointer);
        assert_eq!(list.positional.capacity(), capacity);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn gc_reclaims_datum_field_slack_without_dropping_hot_index() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/gc_slack").unwrap());
        for index in 0..4_096 {
            heap.set_datum_field(
                datum,
                field(&format!("field_{index:04}")),
                Value::number(index as f32),
            )
            .unwrap();
        }
        {
            let datum = heap.datum_mut(datum).unwrap();
            datum.fields.truncate(2_048);
            Arc::make_mut(datum.field_index.as_mut().unwrap())
                .retain(|_, position| *position < 2_048);
        }
        let before = heap.datum(datum).unwrap().fields.capacity();

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[datum], &[]);
        let datum = heap.datum(datum).unwrap();
        assert_eq!(stats.datum_storage.shrunk_field_vectors, 1);
        assert!(stats.datum_storage.reclaimed_capacity_bytes > 0);
        assert_eq!(stats.datum_storage.field_indexes, 1);
        assert_eq!(stats.datum_storage.shrunk_field_indexes, 1);
        assert!(stats.datum_storage.reclaimed_field_index_capacity > 0);
        assert!(datum.fields.capacity() < before);
        assert!(
            datum.field_index.is_some(),
            "hot field reads remain indexed"
        );
        assert_eq!(datum.field(&field("field_0000")), Ok(&Value::number(0.0)),);
        assert_eq!(
            datum.field(&field("field_2047")),
            Ok(&Value::number(2_047.0)),
        );
        assert_eq!(
            datum.field(&field("field_2048")),
            Err(ValueError::MissingField(field("field_2048"))),
        );
    }

    #[test]
    fn initialized_datums_share_layouts_before_gc_and_detach_on_growth() {
        let mut heap = ValueHeap::new();
        let path = TypePath::parse("/datum/birth_layout").unwrap();
        let mut defaults = DatumDefaults::new(path.clone());
        for index in 0..32 {
            defaults.set(
                field(&format!("field_{index:02}")),
                Value::number(index as f32),
            );
        }

        let left = heap.allocate_datum_with_defaults(path.clone(), &[defaults.clone()]);
        let right = heap.allocate_datum_with_defaults(path, &[defaults]);
        let left_datum = heap.datum(left).unwrap();
        let right_datum = heap.datum(right).unwrap();
        let DatumFields::Shared {
            names: left_names, ..
        } = &left_datum.fields
        else {
            panic!("initialized fields should be compact before GC")
        };
        let DatumFields::Shared {
            names: right_names, ..
        } = &right_datum.fields
        else {
            panic!("repeated initialized fields should be compact before GC")
        };
        assert!(Arc::ptr_eq(left_names, right_names));
        assert!(Arc::ptr_eq(
            left_datum.field_index.as_ref().unwrap(),
            right_datum.field_index.as_ref().unwrap(),
        ));

        heap.set_datum_field(left, field("field_07"), Value::number(700.0))
            .unwrap();
        assert!(Arc::ptr_eq(
            heap.datum(left).unwrap().field_index.as_ref().unwrap(),
            heap.datum(right).unwrap().field_index.as_ref().unwrap(),
        ));
        heap.set_datum_field(left, field("dynamic"), Value::number(1.0))
            .unwrap();
        assert!(!Arc::ptr_eq(
            heap.datum(left).unwrap().field_index.as_ref().unwrap(),
            heap.datum(right).unwrap().field_index.as_ref().unwrap(),
        ));
        assert_eq!(
            heap.datum(right).unwrap().field(&field("field_07")),
            Ok(&Value::number(7.0)),
        );
    }

    #[test]
    fn gc_shares_identical_field_layout_indexes_and_detaches_only_for_layout_changes() {
        let mut heap = ValueHeap::new();
        let left = heap.allocate_datum(TypePath::parse("/datum/layout_left").unwrap());
        let right = heap.allocate_datum(TypePath::parse("/datum/layout_right").unwrap());
        for index in 0..64 {
            let name = field(&format!("field_{index:02}"));
            heap.set_datum_field(left, name.clone(), Value::number(index as f32))
                .unwrap();
            heap.set_datum_field(right, name, Value::number((index + 1) as f32))
                .unwrap();
        }
        assert!(!Arc::ptr_eq(
            heap.datum(left).unwrap().field_index.as_ref().unwrap(),
            heap.datum(right).unwrap().field_index.as_ref().unwrap(),
        ));

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[left, right], &[]);
        let shared = Arc::clone(heap.datum(left).unwrap().field_index.as_ref().unwrap());
        assert!(Arc::ptr_eq(
            &shared,
            heap.datum(right).unwrap().field_index.as_ref().unwrap(),
        ));
        assert_eq!(stats.datum_storage.field_indexes, 2);
        assert_eq!(stats.datum_storage.physical_field_indexes, 1);
        assert_eq!(stats.datum_storage.deduplicated_field_indexes, 1);
        assert_eq!(stats.datum_storage.field_index_fingerprints_computed, 2);
        assert_eq!(stats.datum_storage.field_index_pointer_cache_hits, 0);
        assert_eq!(stats.datum_storage.field_index_exact_layout_comparisons, 1);
        assert!(stats.datum_storage.deduplicated_field_index_bytes > 0);
        assert_eq!(stats.datum_storage.shared_field_name_datums, 2);
        assert_eq!(stats.datum_storage.shared_field_name_logical_slots, 128);
        assert_eq!(stats.datum_storage.shared_field_name_layouts, 1);
        assert_eq!(stats.datum_storage.shared_field_name_physical_slots, 64);
        assert_eq!(
            stats.datum_storage.shared_field_name_bytes_saved,
            64 * std::mem::size_of::<FieldName>(),
        );
        let DatumFields::Shared {
            names: left_names, ..
        } = &heap.datum(left).unwrap().fields
        else {
            panic!("left layout should be compacted")
        };
        let DatumFields::Shared {
            names: right_names, ..
        } = &heap.datum(right).unwrap().fields
        else {
            panic!("right layout should be compacted")
        };
        assert!(Arc::ptr_eq(left_names, right_names));
        assert_eq!(
            heap.datum(left).unwrap().field(&field("field_07")),
            Ok(&Value::number(7.0)),
        );
        assert_eq!(
            heap.datum(right).unwrap().field(&field("field_07")),
            Ok(&Value::number(8.0)),
        );

        // Updating an existing value does not mutate the layout index, so the
        // shared O(1) cache stays shared.
        heap.set_datum_field(left, field("field_07"), Value::number(700.0))
            .unwrap();
        assert!(Arc::ptr_eq(
            &shared,
            heap.datum(left).unwrap().field_index.as_ref().unwrap(),
        ));

        // Adding a new field changes one layout and therefore detaches only
        // that datum's index through Arc::make_mut.
        heap.set_datum_field(left, field("dynamic_field"), Value::number(1.0))
            .unwrap();
        assert!(!Arc::ptr_eq(
            heap.datum(left).unwrap().field_index.as_ref().unwrap(),
            heap.datum(right).unwrap().field_index.as_ref().unwrap(),
        ));
        assert_eq!(
            heap.datum(left).unwrap().field(&field("dynamic_field")),
            Ok(&Value::number(1.0)),
        );
        assert_eq!(
            heap.datum(right).unwrap().field(&field("dynamic_field")),
            Err(ValueError::MissingField(field("dynamic_field"))),
        );

        // Returning the detached datum to the old layout lets the next GC
        // merge it back into the canonical shared index. A later collection
        // fingerprints that physical Arc once and resolves the other logical
        // datum through the pointer cache without comparing every field.
        drop(shared);
        assert_eq!(
            heap.delete_datum_field(left, &field("dynamic_field"))
                .unwrap(),
            Some(Value::number(1.0)),
        );
        let merged = heap.collect_unreachable_values_from_ids_with_stats(&[left, right], &[]);
        assert!(Arc::ptr_eq(
            heap.datum(left).unwrap().field_index.as_ref().unwrap(),
            heap.datum(right).unwrap().field_index.as_ref().unwrap(),
        ));
        assert_eq!(merged.datum_storage.deduplicated_field_indexes, 1);
        assert_eq!(merged.datum_storage.field_index_fingerprints_computed, 2);
        assert_eq!(merged.datum_storage.field_index_pointer_cache_hits, 0);
        assert_eq!(merged.datum_storage.field_index_exact_layout_comparisons, 1,);

        let repeated = heap.collect_unreachable_values_from_ids_with_stats(&[left, right], &[]);
        assert_eq!(repeated.datum_storage.deduplicated_field_indexes, 0);
        assert_eq!(repeated.datum_storage.field_index_fingerprints_computed, 1);
        assert_eq!(repeated.datum_storage.field_index_pointer_cache_hits, 1);
        assert_eq!(
            repeated.datum_storage.field_index_exact_layout_comparisons,
            0,
        );
    }

    #[test]
    fn gc_shares_small_linear_field_names_and_detaches_on_layout_growth() {
        let mut heap = ValueHeap::new();
        let left = heap.allocate_datum(TypePath::parse("/datum/small_left").unwrap());
        let right = heap.allocate_datum(TypePath::parse("/datum/small_right").unwrap());
        for datum in [left, right] {
            heap.set_datum_field(datum, field("name"), Value::text("value"))
                .unwrap();
            heap.set_datum_field(datum, field("count"), Value::number(2.0))
                .unwrap();
        }
        let owned_snapshot = heap.datum(left).unwrap().clone();

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[left, right], &[]);
        assert_eq!(heap.datum(left).unwrap(), &owned_snapshot);
        assert_eq!(stats.datum_storage.shared_field_name_datums, 2);
        assert_eq!(stats.datum_storage.shared_field_name_layouts, 1);
        assert_eq!(stats.datum_storage.shared_field_name_logical_slots, 4);
        assert_eq!(stats.datum_storage.shared_field_name_physical_slots, 2);
        let DatumFields::Shared {
            names: left_names, ..
        } = &heap.datum(left).unwrap().fields
        else {
            panic!("left layout should be compacted")
        };
        let DatumFields::Shared {
            names: right_names, ..
        } = &heap.datum(right).unwrap().fields
        else {
            panic!("right layout should be compacted")
        };
        assert!(Arc::ptr_eq(left_names, right_names));

        heap.set_datum_field(left, field("extra"), Value::number(3.0))
            .unwrap();
        assert_eq!(
            heap.datum(left).unwrap().field(&field("extra")),
            Ok(&Value::number(3.0)),
        );
        assert_eq!(
            heap.datum(right).unwrap().field(&field("extra")),
            Err(ValueError::MissingField(field("extra"))),
        );
    }

    #[test]
    fn gc_field_index_sharing_never_merges_different_field_orders() {
        let mut heap = ValueHeap::new();
        let left = heap.allocate_datum(TypePath::parse("/datum/layout_order_left").unwrap());
        let right = heap.allocate_datum(TypePath::parse("/datum/layout_order_right").unwrap());
        let names = (0..16)
            .map(|index| field(&format!("field_{index:02}")))
            .collect::<Vec<_>>();
        for (index, name) in names.iter().enumerate() {
            heap.set_datum_field(left, name.clone(), Value::number(index as f32))
                .unwrap();
        }
        for (index, name) in names.iter().rev().enumerate() {
            heap.set_datum_field(right, name.clone(), Value::number(index as f32))
                .unwrap();
        }

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[left, right], &[]);
        assert!(!Arc::ptr_eq(
            heap.datum(left).unwrap().field_index.as_ref().unwrap(),
            heap.datum(right).unwrap().field_index.as_ref().unwrap(),
        ));
        assert_eq!(stats.datum_storage.physical_field_indexes, 2);
        assert_eq!(stats.datum_storage.deduplicated_field_indexes, 0);
        assert_eq!(
            heap.datum(left).unwrap().fields().next().unwrap().0,
            &names[0],
        );
        assert_eq!(
            heap.datum(right).unwrap().fields().next().unwrap().0,
            &names[15],
        );
    }

    #[test]
    fn gc_reclaims_distributed_field_index_slack_below_old_floor() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/gc_index_slack").unwrap());
        for index in 0..64 {
            heap.set_datum_field(
                datum,
                field(&format!("field_{index:02}")),
                Value::number(index as f32),
            )
            .unwrap();
        }
        {
            let datum = heap.datum_mut(datum).unwrap();
            datum.fields.truncate(52);
            Arc::make_mut(datum.field_index.as_mut().unwrap()).retain(|_, position| *position < 52);
        }
        let before = heap
            .datum(datum)
            .unwrap()
            .field_index
            .as_ref()
            .unwrap()
            .capacity();
        assert!(before.saturating_sub(52) < 128);

        let stats = heap.collect_unreachable_values_from_ids_with_stats(&[datum], &[]);
        let datum_record = heap.datum(datum).unwrap();
        let after = datum_record.field_index.as_ref().unwrap().capacity();
        let reclaimed = before.saturating_sub(after);
        assert!(reclaimed >= 8);
        assert_eq!(stats.datum_storage.shrunk_field_indexes, 1);
        assert_eq!(
            stats.datum_storage.reclaimed_field_index_capacity,
            reclaimed,
        );
        assert_eq!(
            stats.datum_storage.reclaimed_field_index_bytes,
            reclaimed.saturating_mul(std::mem::size_of::<(FieldName, usize)>()),
        );
        assert!(after >= 56, "the compacted index retains 6.25% headroom");
        assert_eq!(
            datum_record.field(&field("field_51")),
            Ok(&Value::number(51.0)),
        );

        let retained_capacity = after;
        for index in 64..68 {
            heap.set_datum_field(
                datum,
                field(&format!("field_{index:02}")),
                Value::number(index as f32),
            )
            .unwrap();
        }
        assert_eq!(
            heap.datum(datum)
                .unwrap()
                .field_index
                .as_ref()
                .unwrap()
                .capacity(),
            retained_capacity,
            "6.25% immediate growth must not reallocate the index",
        );
    }

    #[test]
    fn bulk_subtraction_removes_last_multiset_occurrences_in_one_compaction() {
        let mut list = DmList::default();
        for value in ["a", "b", "a", "a", "c"] {
            list.add(text(value));
        }
        let snapshot = list.clone();
        assert_eq!(
            list.subtract_entries(&[text("a"), text("b"), text("a")]),
            Ok(3)
        );
        assert_eq!(list.len(), 2);
        assert_eq!(list.get(1).unwrap(), &text("a"));
        assert_eq!(list.get(2).unwrap(), &text("c"));
        assert_eq!(snapshot.len(), 5, "COW source remains unchanged");
        assert_eq!(snapshot.get(4).unwrap(), &text("a"));

        let shared = list.clone();
        assert_eq!(list.subtract_entries(&[text("missing")]), Ok(0));
        assert!(Arc::ptr_eq(
            list.storage.as_ref().unwrap(),
            shared.storage.as_ref().unwrap()
        ));
    }

    #[test]
    fn bulk_subtraction_preserves_mixed_associations_and_fallback_equality() {
        let mut list = DmList::default();
        list.add(text("x"));
        list.set_key(text("key"), text("associated"));
        list.add(text("key"));
        let modified = Value::ModifiedTypePath(Arc::new(ModifiedTypePath::new(
            TypePath::parse("/obj/item").unwrap(),
            vec![(field("amount"), Value::number(3.0))],
        )));
        list.add(modified.clone());
        list.set_key(text("keep"), text("value"));

        assert_eq!(
            list.subtract_entries(&[text("key"), modified.clone()]),
            Ok(2)
        );
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(1).unwrap(), &text("x"));
        assert_eq!(list.get(2).unwrap(), &text("key"));
        assert_eq!(list.get(3).unwrap(), &text("keep"));
        assert_eq!(list.get_key(&text("key")).unwrap(), &text("associated"));
        assert_eq!(list.get_key(&text("keep")).unwrap(), &text("value"));
    }

    #[test]
    fn bulk_subtraction_compacts_a_lazy_prefix_and_self_snapshot_exactly() {
        let mut list = DmList::default();
        for value in 0..128 {
            list.add(Value::number(value as f32));
        }
        list.cut_range(1, 65).unwrap();
        let rhs = list
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(list.subtract_entries(&rhs), Ok(64));
        assert!(list.is_empty());
        assert_eq!(list.prefix_head, 0);
        list.add(text("reused"));
        assert_eq!(list.get(1).unwrap(), &text("reused"));
    }

    #[test]
    fn live_heap_counts_track_allocation_reuse_and_destruction() {
        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        let datum = heap.allocate_datum(TypePath::parse("/datum/example").unwrap());
        assert_eq!(heap.live_list_count(), 1);
        assert_eq!(heap.live_datum_count(), 1);
        heap.destroy_list(list).unwrap();
        heap.destroy_datum(datum).unwrap();
        assert_eq!(heap.live_list_count(), 0);
        assert_eq!(heap.live_datum_count(), 0);
        let reused = heap.allocate_list();
        assert_ne!(reused, list);
        assert_eq!(heap.live_list_count(), 1);
    }

    #[test]
    fn arena_stats_track_chunks_free_slots_and_reuse_in_constant_time() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/example").unwrap());
        assert_eq!(
            heap.datum_arena_stats(),
            ArenaStats {
                live: 1,
                slots: 1,
                free: 0,
                chunks: 1,
                reserved: ARENA_CHUNK_SLOTS,
            }
        );

        let lists = (0..=ARENA_CHUNK_SLOTS)
            .map(|_| heap.allocate_list())
            .collect::<Vec<_>>();
        assert_eq!(
            heap.list_arena_stats(),
            ArenaStats {
                live: ARENA_CHUNK_SLOTS + 1,
                slots: ARENA_CHUNK_SLOTS + 1,
                free: 0,
                chunks: 2,
                reserved: ARENA_CHUNK_SLOTS * 2,
            }
        );
        heap.destroy_list(lists[0]).unwrap();
        assert_eq!(heap.list_arena_stats().free, 1);
        let replacement = heap.allocate_list();
        assert_eq!(replacement.index(), lists[0].index());
        let reused = heap.list_arena_stats();
        assert_eq!(reused.live, ARENA_CHUNK_SLOTS + 1);
        assert_eq!(reused.slots, ARENA_CHUNK_SLOTS + 1);
        assert_eq!(reused.free, 0);
        assert_eq!(reused.chunks, 2);

        heap.destroy_datum(datum).unwrap();
        let datum_stats = heap.datum_arena_stats();
        assert_eq!(datum_stats.live, 0);
        assert_eq!(datum_stats.slots, 1);
        assert_eq!(datum_stats.free, 1);
    }

    #[test]
    fn list_collection_preserves_recursive_roots_and_live_datum_fields() {
        let mut heap = ValueHeap::new();
        let frame_root = heap.allocate_list();
        let nested = heap.allocate_list();
        let datum_root = heap.allocate_list();
        let garbage = heap.allocate_list();
        heap.list_mut(frame_root)
            .unwrap()
            .set_key(text("nested"), Value::List(nested));
        heap.list_mut(nested).unwrap().add(Value::List(frame_root));
        let datum = heap.allocate_datum(TypePath::parse("/datum/test").unwrap());
        heap.set_datum_field(datum, field("owned"), Value::List(datum_root))
            .unwrap();

        assert_eq!(
            heap.collect_unreachable_lists(&[Value::List(frame_root)]),
            1
        );
        assert!(heap.list(frame_root).is_ok());
        assert!(heap.list(nested).is_ok());
        assert!(heap.list(datum_root).is_ok());
        assert!(matches!(
            heap.list(garbage),
            Err(ValueError::StaleList(id)) if id == garbage
        ));

        let reused = heap.allocate_list();
        assert_eq!(reused.index(), garbage.index());
        assert_ne!(reused.generation(), garbage.generation());
    }

    #[test]
    fn value_collection_reclaims_unreachable_datum_list_cycles() {
        let mut heap = ValueHeap::new();
        let root = heap.allocate_datum(TypePath::parse("/datum/root").unwrap());
        let child = heap.allocate_datum(TypePath::parse("/datum/child").unwrap());
        let reachable = heap.allocate_list();
        heap.set_datum_field(root, field("items"), Value::List(reachable))
            .unwrap();
        heap.list_mut(reachable).unwrap().add(Value::Datum(child));
        heap.set_datum_field(child, field("owner"), Value::Datum(root))
            .unwrap();

        let garbage_datum = heap.allocate_datum(TypePath::parse("/datum/garbage").unwrap());
        let garbage_list = heap.allocate_list();
        heap.set_datum_field(garbage_datum, field("cycle"), Value::List(garbage_list))
            .unwrap();
        heap.list_mut(garbage_list)
            .unwrap()
            .add(Value::Datum(garbage_datum));

        assert_eq!(
            heap.collect_unreachable_values(&[Value::Datum(root)]),
            (1, 1)
        );
        assert!(heap.datum(root).is_ok());
        assert!(heap.datum(child).is_ok());
        assert!(heap.list(reachable).is_ok());
        assert!(matches!(
            heap.datum(garbage_datum),
            Err(ValueError::StaleDatum(id)) if id == garbage_datum
        ));
        assert!(matches!(
            heap.list(garbage_list),
            Err(ValueError::StaleList(id)) if id == garbage_list
        ));
    }

    #[test]
    fn compact_identity_collection_deduplicates_dense_roots_and_validates_generations() {
        let mut heap = ValueHeap::new();
        let root = heap.allocate_datum(TypePath::parse("/datum/root").unwrap());
        let held = heap.allocate_list();
        heap.set_datum_field(root, field("held"), Value::List(held))
            .unwrap();
        let stale = heap.allocate_datum(TypePath::parse("/datum/stale").unwrap());
        heap.destroy_datum(stale).unwrap();
        let replacement = heap.allocate_datum(TypePath::parse("/datum/replacement").unwrap());
        assert_eq!(stale.index(), replacement.index());
        let garbage = heap.allocate_list();

        let datum_roots = std::iter::repeat_n(root, 100_000)
            .chain([stale])
            .collect::<Vec<_>>();
        assert_eq!(
            heap.collect_unreachable_values_from_ids(&datum_roots, &[]),
            (1, 1),
        );
        assert!(heap.datum(root).is_ok());
        assert!(heap.list(held).is_ok());
        assert!(heap.datum(replacement).is_err());
        assert!(heap.list(garbage).is_err());
    }

    #[test]
    fn canonicalize_deleted_datum_and_list_refs_are_null() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/example").unwrap());
        let list = heap.allocate_list();
        let value = Value::Datum(datum);
        heap.destroy_datum(datum).unwrap();
        assert_eq!(heap.canonicalize_value(&value), Value::Null);
        let value = Value::List(list);
        heap.destroy_list(list).unwrap();
        assert_eq!(heap.canonicalize_value(&Value::List(list)), Value::Null);
        assert_eq!(heap.canonicalize_value(&value), Value::Null);
    }

    #[test]
    fn packed_values_are_pointer_free_and_round_trip_exact_bits_and_handles() {
        assert_eq!(std::mem::size_of::<PackedValue>(), 8);
        let values = [
            Value::Null,
            Value::Number(DmNumberBits::from_f32(f32::from_bits(0x8000_0000))),
            Value::Number(DmNumberBits::from_f32(f32::from_bits(0x7fc1_2345))),
            Value::Datum(DatumId::from_parts(
                PACKED_HANDLE_COMPONENT_MASK,
                PACKED_HANDLE_COMPONENT_MASK - 1,
            )),
            Value::List(ListId::from_parts(
                PACKED_HANDLE_COMPONENT_MASK - 2,
                PACKED_HANDLE_COMPONENT_MASK - 3,
            )),
        ];
        for value in values {
            let restored = PackedValue::try_from_value(&value)
                .expect("pointer-free value should pack")
                .into_value();
            match (&value, &restored) {
                (Value::Number(left), Value::Number(right)) => {
                    assert_eq!(left.bits(), right.bits());
                }
                _ => assert_eq!(value, restored),
            }
        }
        assert_eq!(
            PackedValue::try_from_value(&Value::Datum(DatumId::from_parts(7, 9)))
                .unwrap()
                .heap_handles(),
            (Some(DatumId::from_parts(7, 9)), None)
        );
    }

    #[test]
    fn packed_values_reject_handle_components_that_would_be_truncated() {
        assert!(
            PackedValue::try_from_value(&Value::Datum(DatumId::from_parts(1 << 31, 0))).is_none()
        );
        assert!(
            PackedValue::try_from_value(&Value::List(ListId::from_parts(0, 1 << 31))).is_none()
        );
    }

    #[test]
    fn packed_values_reject_interned_and_owned_variants() {
        let values = [
            Value::text("text"),
            Value::file("asset.dmi"),
            Value::TypePath(TypePath::parse("/datum/example").unwrap()),
            Value::ModifiedTypePath(Arc::new(ModifiedTypePath::new(
                TypePath::parse("/obj/item").unwrap(),
                Vec::new(),
            ))),
        ];
        assert!(
            values
                .iter()
                .all(|value| PackedValue::try_from_value(value).is_none())
        );
    }

    #[test]
    fn canonicalize_deleted_refs_equal_null_and_are_falsy() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(TypePath::parse("/datum/example").unwrap());
        let copied = Value::Datum(datum);
        heap.destroy_datum(datum).unwrap();
        assert!(matches!(heap.canonicalize_value(&copied), Value::Null));
        assert_eq!(heap.truthy(&copied).unwrap(), false);
        assert_eq!(heap.canonicalize_value(&copied), Value::Null);
    }

    #[test]
    fn canonicalized_list_keys_preserve_associative_key_lookup_identity() {
        let mut heap = ValueHeap::new();
        let list = heap.allocate_list();
        let stale_key = heap.allocate_datum(TypePath::parse("/datum/example").unwrap());
        let stored_value = Value::number(7.0);
        heap.list_mut(list)
            .unwrap()
            .set_key(Value::Datum(stale_key), stored_value.clone());
        heap.destroy_datum(stale_key).unwrap();
        let lookup = Value::Datum(stale_key);
        assert_eq!(heap.list(list).unwrap().get_key(&lookup), Ok(&stored_value));
    }

    #[test]
    fn positional_reservation_preserves_list_semantics() {
        let mut list = DmList::default();
        list.reserve_positional(4);
        list.add(Value::number(1.0));
        list.add(Value::number(2.0));
        assert_eq!(list.len(), 2);
        assert_eq!(list.get(1), Ok(&Value::number(1.0)));
        assert_eq!(list.get(2), Ok(&Value::number(2.0)));
    }

    #[test]
    fn bulk_positional_extend_matches_add_and_detaches_shared_storage_once() {
        let mut bulk = DmList::default();
        bulk.add(Value::number(1.0));
        bulk.set_key(Value::text("key"), Value::number(7.0));
        let shared_before_extend = bulk.clone();
        assert_eq!(
            bulk.extend_positional([Value::number(2.0), Value::number(3.0), Value::number(4.0),]),
            5
        );

        let mut repeated = shared_before_extend.clone();
        for value in [2.0, 3.0, 4.0] {
            repeated.add(Value::number(value));
        }
        assert_eq!(bulk.len(), repeated.len());
        for index in 1..=bulk.len() {
            assert!(
                bulk.get(index)
                    .unwrap()
                    .semantic_eq(repeated.get(index).unwrap())
            );
        }
        assert_eq!(
            bulk.get_key(&Value::text("key")),
            repeated.get_key(&Value::text("key"))
        );
        assert_eq!(shared_before_extend.len(), 2);
        assert_eq!(shared_before_extend.get(1), Ok(&Value::number(1.0)));
        assert_eq!(shared_before_extend.get(2), Ok(&Value::text("key")));
    }

    #[test]
    fn heap_snapshot_roundtrip_preserves_handles_aliases_and_free_order() {
        let mut heap = ValueHeap::new();
        let discarded = heap.allocate_datum(TypePath::parse("/datum/discarded").unwrap());
        let owner = heap.allocate_datum(TypePath::parse("/datum/owner").unwrap());
        let values = heap.allocate_list();
        heap.list_mut(values).unwrap().add(Value::Datum(owner));
        heap.list_mut(values)
            .unwrap()
            .set_key(Value::text("answer"), Value::number(42.0));
        heap.list_mut(values).unwrap().add(Value::List(values));
        heap.set_datum_field(
            owner,
            FieldName::parse("values").unwrap(),
            Value::List(values),
        )
        .unwrap();
        heap.set_datum_field(
            owner,
            FieldName::parse("kind").unwrap(),
            Value::ModifiedTypePath(Arc::new(ModifiedTypePath::new(
                TypePath::parse("/obj/item").unwrap(),
                vec![(FieldName::parse("amount").unwrap(), Value::number(15.0))],
            ))),
        )
        .unwrap();
        heap.destroy_datum(discarded).unwrap();

        let snapshot = heap.snapshot();
        let mut restored = ValueHeap::from_snapshot(snapshot.clone()).unwrap();
        assert_eq!(restored.snapshot(), snapshot);
        assert_eq!(
            restored.datum_field(owner, &FieldName::parse("values").unwrap()),
            Ok(&Value::List(values))
        );
        assert_eq!(
            restored.list(values).unwrap().get(1),
            Ok(&Value::Datum(owner))
        );
        assert_eq!(
            restored
                .list(values)
                .unwrap()
                .get_key(&Value::text("answer")),
            Ok(&Value::number(42.0))
        );
        assert_eq!(
            restored.list(values).unwrap().get(3),
            Ok(&Value::List(values))
        );

        let reused = restored.allocate_datum(TypePath::parse("/datum/reused").unwrap());
        assert_eq!(reused.index(), discarded.index());
        assert_eq!(reused.generation(), discarded.generation() + 1);
    }

    #[test]
    fn heap_snapshot_rejects_inconsistent_free_list() {
        let mut heap = ValueHeap::new();
        let live = heap.allocate_list();
        let mut snapshot = heap.snapshot();
        snapshot.list_free.push(live.index());
        assert!(matches!(
            ValueHeap::from_snapshot(snapshot),
            Err(ValueError::CorruptHeapSnapshot(_))
        ));
    }

    #[test]
    #[ignore = "release-only full-z block list allocation benchmark"]
    fn positional_reservation_benchmark() {
        const CELLS: usize = 255 * 255 * 2;
        const ROUNDS: usize = 16;
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            let mut list = DmList::default();
            for index in 0..CELLS {
                list.add(Value::number(index as f32));
            }
            std::hint::black_box(list);
        }
        let unreserved = started.elapsed();
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            let mut list = DmList::default();
            list.reserve_positional(CELLS);
            for index in 0..CELLS {
                list.add(Value::number(index as f32));
            }
            std::hint::black_box(list);
        }
        let reserved = started.elapsed();
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            let mut list = DmList::default();
            list.extend_positional((0..CELLS).map(|index| Value::number(index as f32)));
            std::hint::black_box(list);
        }
        let bulk = started.elapsed();
        eprintln!(
            "full-z-list-cells={} rounds={} unreserved_ms={} reserved_ms={} bulk_ms={} bulk_vs_reserved={:.2}",
            CELLS,
            ROUNDS,
            unreserved.as_millis(),
            reserved.as_millis(),
            bulk.as_millis(),
            reserved.as_secs_f64() / bulk.as_secs_f64(),
        );
    }
}
