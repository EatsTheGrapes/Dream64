//! Type metadata and initial-value caches.
//!
//! Split out of `state.rs`: the `ExecutionState` surface that installs the
//! immutable runtime type catalog (paths, parents, subtype intervals) and
//! the compile-time initial-value catalog, plus the derived `initial()` /
//! effective-default lookup caches and native-datum default seeding.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use crate::bytecode::{InstanceInitializer, Module};
use crate::{
    MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES, MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE,
    engine_builtin_initial_fields, engine_builtin_initial_value, engine_root_initial_value,
};
use dm_value::{DatumId, FieldName, TypePath, Value};

use crate::execution::state::ExecutionState;

pub(crate) fn build_type_intervals(
    parents: &BTreeMap<TypePath, Option<TypePath>>,
) -> BTreeMap<TypePath, (u32, u32)> {
    fn visit(
        node: &TypePath,
        children: &BTreeMap<TypePath, Vec<TypePath>>,
        seen: &mut HashSet<TypePath>,
        intervals: &mut BTreeMap<TypePath, (u32, u32)>,
        clock: &mut u32,
    ) {
        if !seen.insert(node.clone()) {
            return;
        }
        let start = *clock;
        *clock = clock.saturating_add(1);
        if let Some(descendants) = children.get(node) {
            for child in descendants {
                visit(child, children, seen, intervals, clock);
            }
        }
        let end = *clock;
        *clock = clock.saturating_add(1);
        intervals.insert(node.clone(), (start, end));
    }

    let mut children = BTreeMap::<TypePath, Vec<TypePath>>::new();
    let mut roots = Vec::new();
    for (path, parent) in parents {
        if let Some(parent) = parent
            && parents.contains_key(parent)
        {
            children
                .entry(parent.clone())
                .or_default()
                .push(path.clone());
        } else {
            roots.push(path.clone());
        }
    }
    let mut seen = HashSet::new();
    let mut intervals = BTreeMap::new();
    let mut clock = 0u32;
    for root in roots {
        visit(&root, &children, &mut seen, &mut intervals, &mut clock);
    }
    // Invalid or cyclic catalogs should retain deterministic behavior instead
    // of silently dropping types from the acceleration index.
    for path in parents.keys() {
        visit(path, &children, &mut seen, &mut intervals, &mut clock);
    }
    intervals
}

impl ExecutionState {
    /// Replaces the canonical type catalog used by `typesof()`.
    pub fn set_type_paths(&mut self, paths: impl IntoIterator<Item = TypePath>) {
        self.type_paths = Arc::new(paths.into_iter().collect());
        self.clear_effective_initial_value_cache();
    }

    /// Replaces the canonical type catalog used by `typesof()` with a shared
    /// immutable catalog.
    ///
    /// Runtime images use this to avoid cloning a project's complete object
    /// tree for every dynamically evaluated initializer.
    pub fn set_shared_type_paths(&mut self, paths: Arc<std::collections::BTreeSet<TypePath>>) {
        self.type_paths = paths;
        self.clear_effective_initial_value_cache();
    }

    /// Iterates the canonical type catalog in lexical path order.
    pub fn type_paths(&self) -> impl Iterator<Item = &TypePath> {
        self.type_paths.iter()
    }

    /// Replaces the runtime type-parent catalog used by subtype and `parent_type` lookups.
    pub fn set_type_parents(&mut self, parents: BTreeMap<TypePath, Option<TypePath>>) {
        self.type_intervals = Arc::new(build_type_intervals(&parents));
        self.type_parents = Arc::new(parents);
        self.dynamic_receiver_targets.clear();
        self.dynamic_callsite_targets.clear();
        self.instance_initializer_plans.clear();
        self.clear_effective_initial_value_cache();
    }

    /// Replaces the runtime type-parent catalog with shared immutable metadata.
    pub fn set_shared_type_parents(&mut self, parents: Arc<BTreeMap<TypePath, Option<TypePath>>>) {
        self.type_intervals = Arc::new(build_type_intervals(&parents));
        self.type_parents = parents;
        self.dynamic_receiver_targets.clear();
        self.dynamic_callsite_targets.clear();
        self.instance_initializer_plans.clear();
        self.clear_effective_initial_value_cache();
    }

    pub(crate) fn subtype_interval(&self, path: &TypePath) -> Option<(u32, u32)> {
        self.type_intervals.get(path).copied()
    }

    /// Replaces effective compile-time initial field values for every runtime type.
    pub fn set_initial_values(&mut self, values: BTreeMap<TypePath, BTreeMap<FieldName, Value>>) {
        self.initial_values = Arc::new(values);
        self.rebuild_initial_value_roots();
        self.clear_effective_initial_value_cache();
    }

    /// Replaces effective initial values with shared immutable metadata.
    pub fn set_shared_initial_values(
        &mut self,
        values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,
    ) {
        self.initial_values = values;
        self.rebuild_initial_value_roots();
        self.clear_effective_initial_value_cache();
    }

    /// Installs inherited reflection names for owner-qualified shared fields.
    pub fn set_shared_fields(
        &mut self,
        fields: Arc<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>>,
    ) {
        self.shared_fields = fields;
        // A newly declared shared/static var can shadow what a field read
        // previously resolved as an instance default.
        self.field_slot_cache.clear();
    }

    /// Installs direct per-type initializer programs used by runtime `new`.
    pub fn set_instance_initializers(
        &mut self,
        initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
        module: Option<Arc<Module>>,
    ) {
        self.instance_initializers = initializers;
        self.instance_initializer_module = module;
        self.instance_initializer_plans.clear();
        self.clear_initial_field_value_cache();
    }

    /// Replaces runtime-new initializer metadata and returns the previous catalog.
    pub fn replace_instance_initializers(
        &mut self,
        initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
    ) -> Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>> {
        let previous = std::mem::replace(&mut self.instance_initializers, initializers);
        self.instance_initializer_plans.clear();
        self.clear_initial_field_value_cache();
        previous
    }

    /// Returns a type's runtime parent when the catalog contains that type.
    #[must_use]
    pub fn type_parent(&self, path: &TypePath) -> Option<&TypePath> {
        self.type_parents.get(path).and_then(Option::as_ref)
    }

    /// Returns one effective compile-time initial value when available.
    #[must_use]
    pub fn initial_value(&self, path: &TypePath, field: &FieldName) -> Option<&Value> {
        let mut current = Some(path);
        while let Some(path) = current {
            if let Some(value) = self
                .initial_values
                .get(path)
                .and_then(|fields| fields.get(field))
            {
                return Some(value);
            }
            current = self.type_parent(path);
        }
        None
    }

    pub(crate) fn inherited_initial_values(&self, path: &TypePath) -> BTreeMap<FieldName, Value> {
        let mut hierarchy = Vec::new();
        let mut current = Some(path);
        while let Some(path) = current {
            hierarchy.push(path);
            current = self.type_parent(path);
        }
        let mut values = BTreeMap::new();
        for path in hierarchy.into_iter().rev() {
            if let Some(direct) = self.initial_values.get(path) {
                values.extend(direct.clone());
            }
        }
        values
    }

    pub(crate) fn clear_effective_initial_value_cache(&mut self) {
        self.effective_initial_value_cache.get_mut().clear();
        self.effective_initial_value_cache_entries.set(0);
        // Field-read `InitialValue` routing hints are derived from these
        // catalogs; drop them whenever the catalog is replaced.
        self.field_slot_cache.clear();
        self.clear_initial_field_value_cache();
    }

    pub(crate) fn clear_initial_field_value_cache(&mut self) {
        self.initial_field_value_cache.clear();
        self.initial_field_value_cache_entries = 0;
    }

    pub(crate) fn effective_initial_value(
        &self,
        path: &TypePath,
        field: &FieldName,
    ) -> Option<Value> {
        let cache = self.effective_initial_value_cache.borrow();
        if let Some(value) = cache.get(path).and_then(|fields| fields.get(field)) {
            self.effective_initial_value_hits
                .set(self.effective_initial_value_hits.get().saturating_add(1));
            return value.clone();
        }
        drop(cache);
        self.effective_initial_value_cold
            .set(self.effective_initial_value_cold.get().saturating_add(1));

        let value = self
            .initial_value(path, field)
            .or_else(|| engine_root_initial_value(self, path, field))
            .cloned()
            .or_else(|| engine_builtin_initial_value(path, field));
        let entry_count = self.effective_initial_value_cache_entries.get();
        if entry_count < MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES {
            let mut cache = self.effective_initial_value_cache.borrow_mut();
            let fields = cache.entry(path.clone()).or_default();
            if fields.len() < MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE {
                fields.insert(field.clone(), value.clone());
                self.effective_initial_value_cache_entries
                    .set(entry_count + 1);
            }
        }
        value
    }

    /// Seeds effective project and engine defaults on a datum allocated by a
    /// native constructor (`image()`, `icon()`, `sound()`, and peers). Native
    /// constructors historically allocated raw heap datums, bypassing the
    /// inherited `/datum` fields that ordinary `new` installs.
    pub(crate) fn seed_native_datum_defaults(
        &mut self,
        datum: DatumId,
        path: &TypePath,
    ) -> Result<(), String> {
        let mut defaults = engine_builtin_initial_fields(path);
        defaults.extend(self.inherited_initial_values(path));
        for (field, value) in defaults {
            if self.heap.datum_field(datum, &field).is_err() {
                self.heap
                    .set_datum_field(datum, field, value)
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }

    /// Returns the shared immutable runtime type catalog.
    #[must_use]
    pub fn shared_type_paths(&self) -> Arc<BTreeSet<TypePath>> {
        Arc::clone(&self.type_paths)
    }

    /// Returns the shared immutable runtime inheritance catalog.
    #[must_use]
    pub fn shared_type_parents(&self) -> Arc<BTreeMap<TypePath, Option<TypePath>>> {
        Arc::clone(&self.type_parents)
    }

    /// Returns the shared immutable direct initial-value catalog.
    #[must_use]
    pub fn shared_initial_values(&self) -> Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>> {
        Arc::clone(&self.initial_values)
    }

    /// Returns the shared immutable reflection field catalog.
    #[must_use]
    pub fn shared_fields(&self) -> Arc<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>> {
        Arc::clone(&self.shared_fields)
    }

    /// Returns linked per-instance initializer actions and their portable module.
    #[must_use]
    pub fn shared_instance_initializers(
        &self,
    ) -> (
        Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
        Option<Arc<Module>>,
    ) {
        (
            Arc::clone(&self.instance_initializers),
            self.instance_initializer_module.clone(),
        )
    }
}
