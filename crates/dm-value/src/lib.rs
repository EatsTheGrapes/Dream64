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

use std::fmt;
use std::sync::Arc;

use dm_core::DmNumberBits;

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
    /// A canonical type path value.
    TypePath(TypePath),
    /// Reference to a heap datum.
    Datum(DatumId),
    /// Reference to a mutable heap list.
    List(ListId),
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

    /// Returns the stored binary32 number when this value is numeric.
    #[must_use]
    pub const fn as_number(&self) -> Option<f32> {
        match self {
            Self::Number(number) => Some(number.to_f32()),
            Self::Null | Self::Text(_) | Self::TypePath(_) | Self::Datum(_) | Self::List(_) => None,
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
            (Self::TypePath(left), Self::TypePath(right)) => left == right,
            (Self::Datum(left), Self::Datum(right)) => left == right,
            (Self::List(left), Self::List(right)) => left == right,
            _ => false,
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
            Self::TypePath(path) => write!(formatter, "{path}"),
            Self::Datum(id) => write!(formatter, "datum({id:?})"),
            Self::List(id) => write!(formatter, "list({id:?})"),
        }
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

/// One datum record retained by the heap.
#[derive(Clone, Debug, PartialEq)]
pub struct Datum {
    type_path: TypePath,
    fields: Vec<(FieldName, Value)>,
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
        self.fields
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| value)
            .ok_or_else(|| ValueError::MissingField(name.clone()))
    }

    /// Inserts or updates a field while retaining first-insertion order.
    pub fn set_field(&mut self, name: FieldName, value: Value) -> Option<Value> {
        set_named_field(&mut self.fields, name, value)
    }

    /// Deletes a materialized field and returns its value when present.
    pub fn delete_field(&mut self, name: &FieldName) -> Option<Value> {
        let index = self
            .fields
            .iter()
            .position(|(candidate, _)| candidate == name)?;
        Some(self.fields.remove(index).1)
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
        self.fields.iter().map(|(name, value)| (name, value))
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
    positional: Vec<Value>,
    associative: Vec<(Value, Value)>,
    order: Vec<ListOrder>,
}

#[derive(Clone, Debug)]
enum ListOrder {
    Positional(usize),
    Associative(Value),
}

impl DmList {
    /// Returns the number of values and associative keys in iteration order.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns the number of entries without an associative value.
    #[must_use]
    pub fn positional_len(&self) -> usize {
        self.positional.len()
    }

    /// Returns the number of associative entries.
    #[must_use]
    pub fn associative_len(&self) -> usize {
        self.associative.len()
    }

    /// Returns whether there are no positional or associative entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Appends a positional value and returns its 1-based index.
    pub fn add(&mut self, value: Value) -> usize {
        self.order
            .push(ListOrder::Positional(self.positional.len()));
        self.positional.push(value);
        self.order.len()
    }

    /// Reads a 1-based positional entry.
    ///
    /// # Errors
    ///
    /// Returns a precise index error for zero or an index beyond `len`.
    pub fn get(&self, index: usize) -> Result<&Value, ValueError> {
        let zero_based = checked_index(index, self.order.len())?;
        Ok(match &self.order[zero_based] {
            ListOrder::Positional(position) => &self.positional[*position],
            ListOrder::Associative(key) => key,
        })
    }

    /// Replaces a 1-based positional entry and returns its previous value.
    ///
    /// # Errors
    ///
    /// Returns a precise index error for zero or an index beyond `len`.
    pub fn set(&mut self, index: usize, value: Value) -> Result<Value, ValueError> {
        let zero_based = checked_index(index, self.order.len())?;
        let ListOrder::Positional(position) = self.order[zero_based] else {
            return Err(ValueError::AssociativeIndexAssignment { index });
        };
        Ok(std::mem::replace(&mut self.positional[position], value))
    }

    /// Removes and returns a 1-based positional entry.
    ///
    /// # Errors
    ///
    /// Returns a precise index error for zero or an index beyond `len`.
    pub fn remove(&mut self, index: usize) -> Result<Value, ValueError> {
        let zero_based = checked_index(index, self.order.len())?;
        match self.order.remove(zero_based) {
            ListOrder::Positional(position) => {
                for entry in &mut self.order {
                    if let ListOrder::Positional(other) = entry
                        && *other > position
                    {
                        *other -= 1;
                    }
                }
                Ok(self.positional.remove(position))
            }
            ListOrder::Associative(key) => {
                let Some(association) = self
                    .associative
                    .iter()
                    .position(|(candidate, _)| candidate.semantic_eq(&key))
                else {
                    return Err(ValueError::CorruptListStorage);
                };
                self.associative.remove(association);
                Ok(key)
            }
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
        checked_boundary(index, self.order.len())?;
        let position = self.positional.len();
        self.positional.push(value);
        self.order
            .insert(index - 1, ListOrder::Positional(position));
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
        checked_boundary(start, self.order.len())?;
        checked_boundary(end, self.order.len())?;
        let mut copy = Self::default();
        if end <= start {
            return Ok(copy);
        }
        for entry in &self.order[start - 1..end - 1] {
            match entry {
                ListOrder::Positional(position) => {
                    copy.add(self.positional[*position].clone());
                }
                ListOrder::Associative(key) => {
                    copy.set_key(key.clone(), self.get_key(key)?.clone());
                }
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
        checked_boundary(start, self.order.len())?;
        checked_boundary(end, self.order.len())?;
        if end <= start {
            return Ok(0);
        }
        let count = end - start;
        for _ in 0..count {
            self.remove(start)?;
        }
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
        checked_boundary(start, self.order.len())?;
        checked_boundary(end, self.order.len())?;
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
        let index = (1..=self.len()).rev().find(|index| {
            self.get(*index)
                .is_ok_and(|candidate| candidate.semantic_eq(value))
        })?;
        self.remove(index).ok()
    }

    /// Swaps two 1-based iteration positions while keeping associative values
    /// attached to their keys.
    ///
    /// # Errors
    ///
    /// Returns an index error when either position is outside the list.
    pub fn swap(&mut self, first: usize, second: usize) -> Result<(), ValueError> {
        let first = checked_index(first, self.order.len())?;
        let second = checked_index(second, self.order.len())?;
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
        self.associative
            .iter()
            .find(|(candidate, _)| candidate.semantic_eq(key))
            .map(|(_, value)| value)
            .ok_or(ValueError::MissingKey)
    }

    /// Inserts or replaces an associative value.
    ///
    /// Replacing a key retains its deterministic insertion position and
    /// returns the old value. New keys return `None`.
    pub fn set_key(&mut self, key: Value, value: Value) -> Option<Value> {
        if let Some((_, current)) = self
            .associative
            .iter_mut()
            .find(|(candidate, _)| candidate.semantic_eq(&key))
        {
            return Some(std::mem::replace(current, value));
        }
        if let Some((order_index, position)) = self.order.iter().enumerate().find_map(|(index, entry)| {
            let ListOrder::Positional(position) = entry else {
                return None;
            };
            self.positional[*position]
                .semantic_eq(&key)
                .then_some((index, *position))
        }) {
            let existing_key = self.positional.remove(position);
            for entry in &mut self.order {
                if let ListOrder::Positional(other) = entry
                    && *other > position
                {
                    *other -= 1;
                }
            }
            self.order[order_index] = ListOrder::Associative(existing_key.clone());
            self.associative.push((existing_key, value));
            return None;
        }
        self.order.push(ListOrder::Associative(key.clone()));
        self.associative.push((key, value));
        None
    }

    /// Returns whether an iteration entry is semantically equal to `value`.
    #[must_use]
    pub fn contains(&self, value: &Value) -> bool {
        self.positions()
            .any(|(_, candidate)| candidate.semantic_eq(value))
    }

    /// Removes an associative key and returns its value.
    pub fn remove_key(&mut self, key: &Value) -> Option<Value> {
        let index = self
            .associative
            .iter()
            .position(|(candidate, _)| candidate.semantic_eq(key))?;
        let order_index = self.order.iter().position(|entry| match entry {
            ListOrder::Associative(candidate) => candidate.semantic_eq(key),
            ListOrder::Positional(_) => false,
        })?;
        let (_, value) = self.associative.remove(index);
        self.order.remove(order_index);
        Some(value)
    }

    /// Iterates positional entries in ascending 1-based index order.
    #[must_use]
    pub fn positions(&self) -> impl ExactSizeIterator<Item = (usize, &Value)> {
        self.order.iter().enumerate().map(|(index, entry)| {
            let value = match entry {
                ListOrder::Positional(position) => &self.positional[*position],
                ListOrder::Associative(key) => key,
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

struct Arena<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }
}

impl<T> Arena<T> {
    fn insert(&mut self, value: T) -> (u32, u32) {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none());
            slot.value = Some(value);
            return (index, slot.generation);
        }
        let index = u32::try_from(self.slots.len()).expect("heap cannot exceed u32::MAX slots");
        self.slots.push(Slot {
            generation: 0,
            value: Some(value),
        });
        (index, 0)
    }

    fn get(&self, index: u32, generation: u32) -> Option<&T> {
        let slot = self.slots.get(index as usize)?;
        (slot.generation == generation)
            .then_some(slot.value.as_ref())
            .flatten()
    }

    fn get_mut(&mut self, index: u32, generation: u32) -> Option<&mut T> {
        let slot = self.slots.get_mut(index as usize)?;
        (slot.generation == generation)
            .then_some(slot.value.as_mut())
            .flatten()
    }

    fn remove(&mut self, index: u32, generation: u32) -> Option<T> {
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation != generation {
            return None;
        }
        let value = slot.value.take()?;
        if let Some(next_generation) = slot.generation.checked_add(1) {
            slot.generation = next_generation;
            self.free.push(index);
        }
        Some(value)
    }

    fn iter(&self) -> impl Iterator<Item = (u32, u32, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let index = u32::try_from(index).ok()?;
            Some((index, slot.generation, slot.value.as_ref()?))
        })
    }
}

/// Owner of all mutable datum and list identities.
#[derive(Default)]
pub struct ValueHeap {
    datums: Arena<Datum>,
    lists: Arena<DmList>,
}

impl ValueHeap {
    /// Creates an empty heap.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
            fields: Vec::new(),
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
            fields: Vec::new(),
        };
        for layer in defaults {
            datum.apply_defaults(layer);
        }
        let (index, generation) = self.datums.insert(datum);
        DatumId { index, generation }
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

    /// Evaluates DM truth while rejecting stale heap references.
    ///
    /// Null, numeric zero (including signed zero), and empty text are false.
    /// Live datum/list references and type paths are true. NaN is currently
    /// true because it is nonzero; this requires differential confirmation.
    ///
    /// # Errors
    ///
    /// Returns a stale-handle error when a heap reference is no longer live.
    pub fn truthy(&self, value: &Value) -> Result<bool, ValueError> {
        match value {
            Value::Null => Ok(false),
            Value::Number(number) => Ok(number.to_f32() != 0.0),
            Value::Text(text) => Ok(!text.is_empty()),
            Value::TypePath(_) => Ok(true),
            Value::Datum(id) => self.datum(*id).map(|_| true),
            Value::List(id) => self.list(*id).map(|_| true),
        }
    }
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
        assert!(
            heap.list(copy)
                .unwrap()
                .get(1)
                .unwrap()
                .semantic_eq(&Value::List(child))
        );
        heap.list_mut(copy).unwrap().add(text("copy only"));
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
        assert_eq!(
            heap.truthy(&Value::List(list)),
            Err(ValueError::StaleList(list))
        );
        assert_eq!(
            heap.truthy(&Value::Datum(datum)),
            Err(ValueError::StaleDatum(datum))
        );
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
}
