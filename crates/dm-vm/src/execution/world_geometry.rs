//! World geometry: turfs, areas, contents ownership, and visibility lists.
//!
//! Split out of `state.rs`: `ExecutionState` methods that build and mutate
//! the mutable headless map (coordinate/turf/area indexes, `contents`, and
//! the `vis_contents`/`vis_locs` visibility relationship lists).

use crate::bytecode::InstanceInitializer;
use crate::{allocate_initialized_datum, datum_field_or_initial, instance_initializer_plan};
use dm_value::{DatumId, FieldName, ListId, TypePath, Value};

use crate::execution::state::ExecutionState;

impl ExecutionState {
    /// Rebuilds the compact coordinate and contents-owner indexes from the
    /// datums currently present in the heap.
    ///
    /// Hosts that construct a deliberately minimal world (for example a
    /// client-lobby preflight) use this after installing coordinate fields.
    /// Normal map allocation already invokes the same operation during the
    /// runtime-image handoff.
    pub fn rebuild_world_geometry(&mut self) {
        self.world_turfs.clear();
        self.world_turf_lookup.clear();
        self.world_turf_lookup_dimensions = (0, 0, 0);
        self.world_areas.clear();
        self.contents_owners.clear();
        self.vis_contents_owners.clear();
        self.vis_locs_owners.clear();
        let contents = FieldName::parse("contents").expect("built-in contents field");
        let vis_contents = FieldName::parse("vis_contents").expect("built-in vis_contents field");
        let vis_locs = FieldName::parse("vis_locs").expect("built-in vis_locs field");
        for (id, datum) in self.heap.datums() {
            if let Ok(Value::List(list)) = datum.field(&contents) {
                self.contents_owners.insert(*list, id);
            }
            if let Ok(Value::List(list)) = datum.field(&vis_contents) {
                self.vis_contents_owners.insert(*list, id);
            }
            if let Ok(Value::List(list)) = datum.field(&vis_locs) {
                self.vis_locs_owners.insert(*list, id);
            }
        }
        let x = FieldName::parse("x").expect("built-in coordinate field");
        let y = FieldName::parse("y").expect("built-in coordinate field");
        let z = FieldName::parse("z").expect("built-in coordinate field");
        let loc = FieldName::parse("loc").expect("built-in loc field");
        for (id, datum) in self.heap.datums() {
            let path = datum.type_path().as_str();
            if path != "/turf" && !path.starts_with("/turf/") {
                continue;
            }
            let coordinate = [datum.field(&x), datum.field(&y), datum.field(&z)]
                .map(|value| value.ok().and_then(Value::as_number))
                .map(|value| value.filter(|value| value.is_finite() && value.fract() == 0.0));
            let [Some(x), Some(y), Some(z)] = coordinate else {
                continue;
            };
            #[allow(clippy::cast_possible_truncation)]
            let coordinate = (x as i32, y as i32, z as i32);
            self.world_turfs.insert(coordinate, id);
            if let Ok(Value::Datum(area)) = datum.field(&loc) {
                self.world_areas.insert(coordinate, *area);
            }
        }
        let world = self
            .heap
            .datums()
            .find(|(_, datum)| datum.type_path().as_str() == "/world")
            .map(|(id, _)| id);
        let extents = self.world_turfs.keys().fold(None, |extents, &(x, y, z)| {
            Some(
                extents.map_or((x, y, z), |(maxx, maxy, maxz): (i32, i32, i32)| {
                    (maxx.max(x), maxy.max(y), maxz.max(z))
                }),
            )
        });
        self.rebuild_world_turf_lookup();
        if let (Some(world), Some((maxx, maxy, maxz))) = (world, extents) {
            for (name, value) in [("maxx", maxx), ("maxy", maxy), ("maxz", maxz)] {
                let _ = self.heap.set_datum_field(
                    world,
                    FieldName::parse(name).expect("built-in world dimension field"),
                    Value::number(value as f32),
                );
            }
        }
    }

    pub(crate) fn world_dimension(&self, world: DatumId, name: &str) -> Result<i32, String> {
        let value = self
            .heap
            .datum_field(
                world,
                &FieldName::parse(name).expect("built-in world dimension field"),
            )
            .ok()
            .and_then(Value::as_number)
            .unwrap_or(1.0);
        if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > i32::MAX as f32 {
            return Err(format!(
                "world.{name} must be a positive integer, received {value}"
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        Ok(value as i32)
    }

    pub(crate) fn world_type_field(
        &self,
        world: DatumId,
        name: &str,
        fallback: &str,
    ) -> Result<TypePath, String> {
        let field = FieldName::parse(name).expect("built-in world type field");
        // Runtime images deliberately keep unchanged declared fields sparse.
        // World geometry creation must therefore observe the effective DM
        // initial value before applying the engine fallback, just like an
        // ordinary `world.area` or `world.turf` field read. Falling straight
        // back from a missing heap slot created dynamic z-levels as `/area`
        // and `/turf`, ignoring project declarations such as Monke's
        // `/area/space` and `/turf/open/space/basic`.
        match datum_field_or_initial(self, world, &field) {
            Ok(Value::TypePath(path)) => Ok(path),
            Ok(Value::ModifiedTypePath(path)) => Ok(path.base().clone()),
            Ok(Value::Null) | Err(_) => {
                TypePath::parse(fallback).map_err(|error| error.to_string())
            }
            Ok(value) => Err(format!(
                "world.{name} must be a type path, received {value}"
            )),
        }
    }

    pub(crate) fn ensure_contents(&mut self, datum: DatumId) -> Result<ListId, String> {
        let contents = FieldName::parse("contents").expect("built-in contents field");
        if let Ok(Value::List(list)) = self.heap.datum_field(datum, &contents) {
            self.contents_owners.insert(*list, datum);
            return Ok(*list);
        }
        let list = self.heap.allocate_list();
        self.heap
            .set_datum_field(datum, contents, Value::List(list))
            .map_err(|error| error.to_string())?;
        self.contents_owners.insert(list, datum);
        Ok(list)
    }

    pub(crate) fn contents_owner(&self, list: ListId) -> Option<DatumId> {
        self.contents_owners.get(&list).copied()
    }

    pub(crate) fn visibility_owner(&self, list: ListId) -> Option<(DatumId, bool)> {
        self.vis_contents_owners
            .get(&list)
            .copied()
            .map(|owner| (owner, true))
            .or_else(|| {
                self.vis_locs_owners
                    .get(&list)
                    .copied()
                    .map(|owner| (owner, false))
            })
    }

    pub(crate) fn is_visibility_list(&self, list: ListId) -> bool {
        self.visibility_owner(list).is_some()
    }

    pub(crate) fn visibility_list_accepts(&self, value: &Value) -> bool {
        let Value::Datum(datum) = value else {
            return matches!(value, Value::Null);
        };
        self.heap.datum(*datum).is_ok_and(|datum| {
            let path = datum.type_path().as_str();
            path == "/atom"
                || path.starts_with("/atom/")
                || path == "/obj"
                || path.starts_with("/obj/")
                || path == "/mob"
                || path.starts_with("/mob/")
                || path == "/turf"
                || path.starts_with("/turf/")
                || path == "/area"
                || path.starts_with("/area/")
        })
    }

    pub(crate) fn visibility_members(&self, list: ListId) -> Result<Vec<DatumId>, String> {
        Ok(self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::Datum(datum) => Some(*datum),
                _ => None,
            })
            .collect())
    }

    /// Applies one scalar `vis_contents` addition or removal without
    /// normalizing and diffing the complete relationship list.
    ///
    /// Returns `None` for ordinary lists and for `vis_locs`, whose direct
    /// mutation keeps using the general synchronization path. A handled
    /// `vis_contents` mutation returns whether its membership changed.
    pub(crate) fn mutate_vis_contents_scalar(
        &mut self,
        list: ListId,
        value: &Value,
        add: bool,
    ) -> Result<Option<bool>, String> {
        let Some((owner, true)) = self.visibility_owner(list) else {
            return Ok(None);
        };

        if add {
            if matches!(value, Value::Null) {
                return Ok(Some(false));
            }
            if !self.visibility_list_accepts(value) {
                return Err(format!(
                    "visibility lists can only contain atoms, received {value}"
                ));
            }
        }

        let Value::Datum(member) = value else {
            return Ok(Some(false));
        };
        let member_value = Value::Datum(*member);
        let contains = self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .contains(&member_value);
        if contains == add {
            return Ok(Some(false));
        }

        if add {
            self.heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .add(member_value);
        } else {
            self.heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_last(&member_value);
        }

        let reciprocal = self.ensure_visibility_list(*member, false)?;
        let owner_value = Value::Datum(owner);
        let reciprocal = self
            .heap
            .list_mut(reciprocal)
            .map_err(|error| error.to_string())?;
        if add {
            if !reciprocal.contains(&owner_value) {
                reciprocal.add(owner_value);
            }
        } else {
            while reciprocal.remove_last(&owner_value).is_some() {}
        }
        Ok(Some(true))
    }

    pub(crate) fn ensure_visibility_list(
        &mut self,
        datum: DatumId,
        vis_contents: bool,
    ) -> Result<ListId, String> {
        let name = if vis_contents {
            "vis_contents"
        } else {
            "vis_locs"
        };
        let field = FieldName::parse(name).expect("built-in visibility field");
        if let Ok(Value::List(list)) = self.heap.datum_field(datum, &field) {
            let list = *list;
            if vis_contents {
                self.vis_contents_owners.insert(list, datum);
            } else {
                self.vis_locs_owners.insert(list, datum);
            }
            return Ok(list);
        }
        let list = self.heap.allocate_list();
        self.heap
            .set_datum_field(datum, field, Value::List(list))
            .map_err(|error| error.to_string())?;
        if vis_contents {
            self.vis_contents_owners.insert(list, datum);
        } else {
            self.vis_locs_owners.insert(list, datum);
        }
        Ok(list)
    }

    pub(crate) fn synchronize_visibility_list(
        &mut self,
        list: ListId,
        before: &[DatumId],
    ) -> Result<(), String> {
        let Some((owner, is_vis_contents)) = self.visibility_owner(list) else {
            return Ok(());
        };
        let after = self.visibility_members(list)?;
        for removed in before {
            if after.contains(removed) {
                continue;
            }
            let reciprocal = self.ensure_visibility_list(*removed, !is_vis_contents)?;
            let reciprocal = self
                .heap
                .list_mut(reciprocal)
                .map_err(|error| error.to_string())?;
            while reciprocal.remove_last(&Value::Datum(owner)).is_some() {}
        }
        for added in after {
            if before.contains(&added) {
                continue;
            }
            let reciprocal = self.ensure_visibility_list(added, !is_vis_contents)?;
            let reciprocal = self
                .heap
                .list_mut(reciprocal)
                .map_err(|error| error.to_string())?;
            if !reciprocal.contains(&Value::Datum(owner)) {
                reciprocal.add(Value::Datum(owner));
            }
        }
        Ok(())
    }

    pub(crate) fn normalize_and_synchronize_visibility_list(
        &mut self,
        list: ListId,
        before: &[DatumId],
    ) -> Result<(), String> {
        if !self.is_visibility_list(list) {
            return Ok(());
        }
        let values = self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        let mut normalized = Vec::new();
        for value in values {
            if matches!(value, Value::Null) {
                continue;
            }
            if !self.visibility_list_accepts(&value) {
                return Err(format!(
                    "visibility lists can only contain atoms, received {value}"
                ));
            }
            if !normalized
                .iter()
                .any(|existing: &Value| existing.semantic_eq(&value))
            {
                normalized.push(value);
            }
        }
        let target = self
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?;
        target.resize(0).map_err(|error| error.to_string())?;
        for value in normalized {
            target.add(value);
        }
        self.synchronize_visibility_list(list, before)
    }

    pub(crate) fn default_area_for_world(&mut self, world: DatumId) -> Result<DatumId, String> {
        let path = self.world_type_field(world, "area", "/area")?;
        if let Some(area) = self.default_world_area
            && self
                .heap
                .datum(area)
                .is_ok_and(|datum| datum.type_path() == &path)
        {
            return Ok(area);
        }
        let existing = self
            .heap
            .datums()
            .find_map(|(id, datum)| (datum.type_path() == &path).then_some(id));
        let area = match existing {
            Some(area) => area,
            None => allocate_initialized_datum(self, path)?,
        };
        self.ensure_contents(area)?;
        let world_contents = self.ensure_contents(world)?;
        let contents = self
            .heap
            .list_mut(world_contents)
            .map_err(|error| error.to_string())?;
        if !contents.contains(&Value::Datum(area)) {
            contents.add(Value::Datum(area));
        }
        self.default_world_area = Some(area);
        Ok(area)
    }

    pub(crate) fn remove_world_cell(
        &mut self,
        world: DatumId,
        coordinate: (i32, i32, i32),
    ) -> Result<(), String> {
        let Some(turf) = self.world_turfs.remove(&coordinate) else {
            self.world_areas.remove(&coordinate);
            return Ok(());
        };
        if let Some(area) = self.world_areas.remove(&coordinate) {
            let contents = self.ensure_contents(area)?;
            self.heap
                .list_mut(contents)
                .map_err(|error| error.to_string())?
                .remove_first(&Value::Datum(turf));
        }
        let contents = self.ensure_contents(world)?;
        self.heap
            .list_mut(contents)
            .map_err(|error| error.to_string())?
            .remove_first(&Value::Datum(turf));
        self.heap
            .destroy_datum(turf)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub(crate) fn resize_world_geometry(
        &mut self,
        world: DatumId,
        dimensions: (i32, i32, i32),
    ) -> Result<(), String> {
        let (maxx, maxy, maxz) = dimensions;
        if maxx < 1 || maxy < 1 || maxz < 1 {
            return Err("world dimensions must be positive integers".to_owned());
        }
        let removed = self
            .world_turfs
            .keys()
            .copied()
            .filter(|(x, y, z)| *x > maxx || *y > maxy || *z > maxz)
            .collect::<Vec<_>>();
        for coordinate in removed {
            self.remove_world_cell(world, coordinate)?;
        }

        let area = self.default_area_for_world(world)?;
        let turf_path = self.world_type_field(world, "turf", "/turf")?;
        let area_contents = self.ensure_contents(area)?;
        // Constant initializer actions are deliberately omitted for fresh
        // compact engine turfs: their values remain in the shared initial
        // catalog. Only runtime initializer programs can make allocation
        // observable and require the general per-cell path.
        let turf_has_runtime_initializers = instance_initializer_plan(self, &turf_path)
            .iter()
            .any(|initializer| matches!(initializer, InstanceInitializer::Program { .. }));
        let mut bulk_area_members = Vec::new();
        // A full `world.maxz++` slab is thousands of identically-typed turfs.
        // Resolve `world.contents` once so both the compact path and the
        // template-clone path below can append their new cells in one shot.
        let world_contents = self
            .global(&FieldName::parse("world").expect("built-in world global"))
            .and_then(|value| (value == &Value::Datum(world)).then_some(()))
            .and_then(|()| {
                self.heap
                    .datum_field(
                        world,
                        &FieldName::parse("contents").expect("contents field"),
                    )
                    .ok()
            })
            .and_then(|value| match value {
                Value::List(list) => Some(*list),
                _ => None,
            });
        let mut bulk_world_members = Vec::new();
        let coordinate_fields = ["x", "y", "z"]
            .map(|name| FieldName::parse(name).expect("built-in coordinate field is valid"));
        let loc = FieldName::parse("loc").expect("built-in loc field is valid");
        // `world.maxz` (and friends) are grown one slab at a time by DM engines,
        // so this loop is re-entered many times over the same coordinate space.
        // Consult the flattened lookup grid for the existence probe (O(1), and
        // exactly equivalent to `world_turfs.contains_key` for in-bounds
        // coordinates, as `turf_at` relies on) so an incremental grow pays only
        // for its genuinely new cells.
        let (lookup_maxx, lookup_maxy, lookup_maxz) = self.world_turf_lookup_dimensions;
        let cell_exists = |state: &Self, x: i32, y: i32, z: i32| -> bool {
            if x >= 1
                && y >= 1
                && z >= 1
                && x <= lookup_maxx
                && y <= lookup_maxy
                && z <= lookup_maxz
            {
                let index = ((z - 1) as usize * lookup_maxy as usize + (y - 1) as usize)
                    * lookup_maxx as usize
                    + (x - 1) as usize;
                if let Some(slot) = state.world_turf_lookup.get(index) {
                    return slot.is_some();
                }
            }
            state.world_turfs.contains_key(&(x, y, z))
        };
        // Every turf in a fresh slab is the same type initialized in the same
        // (src-independent) context, so its runtime initializer programs
        // otherwise re-run their bytecode tens of thousands of times to compute
        // the identical field set. Run them once for a template cell, then
        // replay the resulting fields onto its siblings (deep-copying any list
        // value so instances never alias mutable state). A datum-valued
        // initializer result is not safely shareable, so fall back to the
        // per-cell path if one appears.
        let mut template_fields: Option<Vec<(FieldName, Value)>> = None;
        let mut template_shareable = true;
        for z in 1..=maxz {
            for y in 1..=maxy {
                for x in 1..=maxx {
                    let coordinate = (x, y, z);
                    if cell_exists(self, x, y, z) {
                        continue;
                    }
                    let turf = if !turf_has_runtime_initializers {
                        self.heap.allocate_datum(turf_path.clone())
                    } else if let Some(fields) =
                        template_fields.as_ref().filter(|_| template_shareable)
                    {
                        let turf = self.heap.allocate_datum(turf_path.clone());
                        for (name, value) in fields {
                            let value = match value {
                                Value::List(list) => Value::List(
                                    self.heap
                                        .copy_list(*list)
                                        .map_err(|error| error.to_string())?,
                                ),
                                other => other.clone(),
                            };
                            self.heap
                                .set_datum_field(turf, name.clone(), value)
                                .map_err(|error| error.to_string())?;
                        }
                        if world_contents.is_some() {
                            bulk_world_members.push(Value::Datum(turf));
                        }
                        turf
                    } else {
                        // `allocate_initialized_datum` adds the cell to
                        // `world.contents` itself, so it is never pushed to the
                        // bulk list below.
                        let turf = allocate_initialized_datum(self, turf_path.clone())?;
                        if template_fields.is_none() {
                            let snapshot = self
                                .heap
                                .datum_fields(turf)
                                .map_err(|error| error.to_string())?
                                .filter(|(name, _)| {
                                    !coordinate_fields.iter().any(|field| field == *name)
                                        && *name != &loc
                                })
                                .map(|(name, value)| (name.clone(), value.clone()))
                                .collect::<Vec<_>>();
                            template_shareable = snapshot
                                .iter()
                                .all(|(_, value)| !matches!(value, Value::Datum(_)));
                            template_fields = Some(snapshot);
                        }
                        turf
                    };
                    for (field, value) in coordinate_fields.iter().zip([x, y, z]) {
                        self.heap
                            .set_datum_field(turf, field.clone(), Value::number(value as f32))
                            .map_err(|error| error.to_string())?;
                    }
                    self.heap
                        .set_datum_field(turf, loc.clone(), Value::Datum(area))
                        .map_err(|error| error.to_string())?;
                    bulk_area_members.push(Value::Datum(turf));
                    if !turf_has_runtime_initializers && world_contents.is_some() {
                        bulk_world_members.push(Value::Datum(turf));
                    }
                    self.world_turfs.insert(coordinate, turf);
                    self.world_areas.insert(coordinate, area);
                }
            }
        }
        if !bulk_area_members.is_empty() {
            self.heap
                .list_mut(area_contents)
                .map_err(|error| error.to_string())?
                .extend_positional(bulk_area_members);
        }
        if let Some(world_contents) = world_contents
            && !bulk_world_members.is_empty()
        {
            self.heap
                .list_mut(world_contents)
                .map_err(|error| error.to_string())?
                .extend_positional(bulk_world_members);
        }
        for (name, value) in [("maxx", maxx), ("maxy", maxy), ("maxz", maxz)] {
            self.heap
                .set_datum_field(
                    world,
                    FieldName::parse(name).expect("built-in world dimension field"),
                    Value::number(value as f32),
                )
                .map_err(|error| error.to_string())?;
        }
        self.rebuild_world_turf_lookup();
        Ok(())
    }

    pub(crate) fn turf_at(&self, x: i32, y: i32, z: i32) -> Option<DatumId> {
        let coordinate = (x, y, z);
        let (maxx, maxy, maxz) = self.world_turf_lookup_dimensions;
        if x >= 1 && y >= 1 && z >= 1 && x <= maxx && y <= maxy && z <= maxz {
            let index = ((z - 1) as usize * maxy as usize + (y - 1) as usize) * maxx as usize
                + (x - 1) as usize;
            if let Some(turf) = self.world_turf_lookup.get(index).copied().flatten() {
                return Some(turf);
            }
        }
        self.world_turfs.get(&coordinate).copied()
    }

    pub(crate) fn rebuild_world_turf_lookup(&mut self) {
        let dimensions = self
            .world_turfs
            .keys()
            .fold((0, 0, 0), |(maxx, maxy, maxz), &(x, y, z)| {
                (maxx.max(x), maxy.max(y), maxz.max(z))
            });
        self.world_turf_lookup_dimensions = dimensions;
        let (maxx, maxy, maxz) = dimensions;
        let Some(length) = usize::try_from(maxx)
            .ok()
            .and_then(|x| usize::try_from(maxy).ok().and_then(|y| x.checked_mul(y)))
            .and_then(|xy| usize::try_from(maxz).ok().and_then(|z| xy.checked_mul(z)))
        else {
            self.world_turf_lookup.clear();
            return;
        };
        self.world_turf_lookup.clear();
        self.world_turf_lookup.resize(length, None);
        for (&(x, y, z), &turf) in &self.world_turfs {
            if x < 1 || y < 1 || z < 1 {
                continue;
            }
            let index = ((z - 1) as usize * maxy as usize + (y - 1) as usize) * maxx as usize
                + (x - 1) as usize;
            if let Some(slot) = self.world_turf_lookup.get_mut(index) {
                *slot = Some(turf);
            }
        }
    }

    pub(crate) fn note_turf_area(&mut self, turf: DatumId, area: DatumId) {
        let coordinate = ["x", "y", "z"]
            .map(|name| FieldName::parse(name).expect("built-in coordinate field"))
            .map(|field| {
                self.heap
                    .datum_field(turf, &field)
                    .ok()
                    .and_then(Value::as_number)
            });
        let [Some(x), Some(y), Some(z)] = coordinate else {
            return;
        };
        if [x, y, z]
            .into_iter()
            .any(|value| !value.is_finite() || value.fract() != 0.0)
        {
            return;
        }
        #[allow(clippy::cast_possible_truncation)]
        self.world_areas
            .insert((x as i32, y as i32, z as i32), area);
    }
}
