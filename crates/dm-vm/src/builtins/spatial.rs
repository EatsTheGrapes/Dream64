//! Spatial indexing, range query, movement, `step`/`walk`/`get_dist`/
//! `turn`/`astype`/`bounds_dist` procedures, and the shared subtype and
//! native-walk advancement helpers used by the scheduler and value layer.

use dm_value::{DatumId, FieldName, ListId, TypePath, Value};
use std::collections::HashSet;

use super::{
    ExecutionState, NativeWalk, NativeWalkKind, builtin_contents_field, builtin_loc_field,
    datum_coordinates, number, runtime_text,
};
/// Adds one turf followed by its direct movable contents, matching the cell
/// ordering used by BYOND/OpenDream's view enumeration. Inventory descendants
/// are not members of the surrounding turf cell and therefore remain hidden.
pub(super) fn append_spatial_cell(
    state: &ExecutionState,
    turf: DatumId,
    seen: &mut HashSet<DatumId>,
    output: &mut Vec<DatumId>,
) {
    // `world_turfs` is the authoritative geometry index. Its keys are updated
    // together with turf allocation/movement, so re-reading x/y/z here only
    // repeats three dynamic field lookups for every cell in every view().
    // Retain the liveness/type check at this boundary so a corrupt or stale
    // handle can never escape through the spatial builtin.
    if !state
        .heap
        .datum(turf)
        .is_ok_and(|datum| super::is_turf_type_path(datum.type_path()))
        || !seen.insert(turf)
    {
        return;
    }
    output.push(turf);

    let contents = builtin_contents_field();
    let members = state
        .heap
        .datum_field(turf, contents)
        .ok()
        .and_then(|value| match value {
            Value::List(list) => state.heap.list(*list).ok(),
            _ => None,
        })
        .map(|list| {
            list.positions()
                .filter_map(|(_, value)| match value {
                    Value::Datum(member) => Some(*member),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for member in members {
        let Ok(datum) = state.heap.datum(member) else {
            continue;
        };
        let path = datum.type_path().as_str();
        if (path == "/area" || path.starts_with("/area/")) || !seen.insert(member) {
            continue;
        }
        output.push(member);
    }
}

pub(super) fn spiral_order_key(
    delta_x: i32,
    delta_y: i32,
    distance_x: i32,
    distance_y: i32,
) -> (u32, u64) {
    let radius = delta_x.unsigned_abs().max(delta_y.unsigned_abs());
    if radius == 0 {
        return (0, 0);
    }
    let radius_i32 = i32::try_from(radius).expect("coordinate delta radius fits i32");
    let vertical_radius = radius_i32.min(distance_y);
    let left_count = if radius_i32 <= distance_x {
        u64::from(vertical_radius.unsigned_abs()) * 2 + 1
    } else {
        0
    };
    if radius_i32 <= distance_x && delta_x == -radius_i32 {
        return (
            radius,
            u64::from((delta_y + vertical_radius).unsigned_abs()),
        );
    }

    let interior_low = (-radius_i32 + 1).max(-distance_x);
    let interior_high = (radius_i32 - 1).min(distance_x);
    let interior_count = if radius_i32 <= distance_y && interior_low <= interior_high {
        u64::from((interior_high - interior_low + 1).unsigned_abs()) * 2
    } else {
        0
    };
    if radius_i32 <= distance_y
        && (delta_y == -radius_i32 || delta_y == radius_i32)
        && delta_x >= interior_low
        && delta_x <= interior_high
    {
        let top = u64::from(delta_y == radius_i32);
        return (
            radius,
            left_count + u64::from((delta_x - interior_low).unsigned_abs()) * 2 + top,
        );
    }

    (
        radius,
        left_count + interior_count + u64::from((delta_y + vertical_radius).unsigned_abs()),
    )
}

pub(super) fn indexed_spatial_candidates(
    state: &ExecutionState,
    center_x: f32,
    center_y: f32,
    center_z: f32,
    distance_x: f32,
    distance_y: f32,
    exclude_center: bool,
) -> Vec<DatumId> {
    let integral_coordinate = |value: f32| -> Option<i32> {
        (value.is_finite()
            && value.fract() == 0.0
            && value >= i32::MIN as f32
            && value <= i32::MAX as f32)
            .then_some(value as i32)
    };
    let (Some(center_x), Some(center_y), Some(center_z)) = (
        integral_coordinate(center_x),
        integral_coordinate(center_y),
        integral_coordinate(center_z),
    ) else {
        return Vec::new();
    };
    let distance_x = distance_x.min(i32::MAX as f32) as i32;
    let distance_y = distance_y.min(i32::MAX as f32) as i32;
    let low_x = center_x.saturating_sub(distance_x);
    let high_x = center_x.saturating_add(distance_x);
    let low_y = center_y.saturating_sub(distance_y);
    let high_y = center_y.saturating_add(distance_y);

    let axis_len = |low: i32, high: i32| {
        u128::try_from(i64::from(high) - i64::from(low) + 1)
            .expect("ordered i32 bounds have a positive span")
    };
    let area = axis_len(low_x, high_x).saturating_mul(axis_len(low_y, high_y));
    let direct_limit = (state.world_turfs.len() as u128)
        .saturating_mul(2)
        .max(4_096);
    let ordered_turfs = if area <= direct_limit {
        let mut turfs = Vec::new();
        if let Some(turf) = state.turf_at(center_x, center_y, center_z) {
            turfs.push(((center_x, center_y), turf));
        }
        for radius in 1..=distance_x.max(distance_y) {
            let vertical_radius = radius.min(distance_y);
            if radius <= distance_x {
                let x = center_x.saturating_sub(radius);
                for delta_y in -vertical_radius..=vertical_radius {
                    if let Some(turf) = state.turf_at(x, center_y.saturating_add(delta_y), center_z)
                    {
                        turfs.push(((x, center_y.saturating_add(delta_y)), turf));
                    }
                }
            }
            if radius <= distance_y {
                let low_delta_x = (-radius + 1).max(-distance_x);
                let high_delta_x = (radius - 1).min(distance_x);
                for delta_x in low_delta_x..=high_delta_x {
                    let x = center_x.saturating_add(delta_x);
                    for delta_y in [-radius, radius] {
                        if let Some(turf) =
                            state.turf_at(x, center_y.saturating_add(delta_y), center_z)
                        {
                            turfs.push(((x, center_y.saturating_add(delta_y)), turf));
                        }
                    }
                }
            }
            if radius <= distance_x {
                let x = center_x.saturating_add(radius);
                for delta_y in -vertical_radius..=vertical_radius {
                    if let Some(turf) = state.turf_at(x, center_y.saturating_add(delta_y), center_z)
                    {
                        turfs.push(((x, center_y.saturating_add(delta_y)), turf));
                    }
                }
            }
        }
        turfs
    } else {
        let mut turfs = state
            .world_turfs
            .iter()
            .filter_map(|((x, y, z), turf)| {
                (*z == center_z && *x >= low_x && *x <= high_x && *y >= low_y && *y <= high_y)
                    .then_some(((*x, *y), *turf))
            })
            .collect::<Vec<_>>();
        turfs.sort_unstable_by_key(|((x, y), _)| {
            spiral_order_key(*x - center_x, *y - center_y, distance_x, distance_y)
        });
        turfs
    };

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for ((x, y), turf) in ordered_turfs {
        if exclude_center && x == center_x && y == center_y {
            continue;
        }
        append_spatial_cell(state, turf, &mut seen, &mut candidates);
    }
    candidates
}

pub(super) fn append_orange_candidate(
    state: &mut ExecutionState,
    output: ListId,
    candidate: DatumId,
    center: &Value,
    loc: &FieldName,
) -> Result<(), String> {
    let datum = state
        .heap
        .datum(candidate)
        .map_err(|error| error.to_string())?;
    if !super::is_atom_type_path(datum.type_path()) || Value::Datum(candidate).semantic_eq(center) {
        return Ok(());
    }
    let candidate_loc =
        super::datum_field_or_initial(state, candidate, loc).map_err(|error| error.to_string())?;
    if candidate_loc.semantic_eq(center) {
        return Ok(());
    }
    state
        .heap
        .list_mut(output)
        .map_err(|error| error.to_string())?
        .add(Value::Datum(candidate));
    Ok(())
}

/// Native form of BYOND's `orange()` using the same indexed cell order as
/// `range()`, but filtering directly into the result. The semantic builtin's
/// historical DM body materialized an intermediate range list and then kept
/// every atom except the center and atoms whose direct `loc` was the center.
pub(super) fn orange_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
    usr: &Value,
) -> Result<Value, String> {
    let output = state.heap.allocate_list();
    let Some(first) = arguments.first() else {
        return Err("orange requires one or two arguments".to_owned());
    };
    let second = arguments.get(1).unwrap_or(usr);
    let (distance, center) = if let Some(distance) = first.as_number() {
        (Some(distance), second)
    } else {
        (second.as_number(), first)
    };
    let Some(distance) = distance else {
        return Ok(Value::List(output));
    };
    if !distance.is_finite() || distance < 0.0 {
        return Ok(Value::List(output));
    }
    let Some((center_x, center_y, center_z)) = datum_coordinates(state, center) else {
        return Ok(Value::List(output));
    };
    let distance = distance.floor();
    let loc = FieldName::parse("loc").expect("built-in loc field");

    if state.world_turfs.is_empty() {
        // Geometry-free fixtures retain range()'s historical arena scan and
        // direct coordinate fields. Production worlds never enter this path.
        let x = FieldName::parse("x").expect("built-in coordinate field");
        let y = FieldName::parse("y").expect("built-in coordinate field");
        let z = FieldName::parse("z").expect("built-in coordinate field");
        let candidates = state
            .heap
            .datums()
            .filter_map(|(candidate, datum)| {
                let path = datum.type_path().as_str();
                if path == "/area" || path.starts_with("/area/") {
                    return None;
                }
                let candidate_x = datum.field(&x).ok()?.as_number()?;
                let candidate_y = datum.field(&y).ok()?.as_number()?;
                let candidate_z = datum.field(&z).ok()?.as_number()?;
                (candidate_z.total_cmp(&center_z).is_eq()
                    && (candidate_x - center_x).abs() <= distance
                    && (candidate_y - center_y).abs() <= distance)
                    .then_some(candidate)
            })
            .collect::<Vec<_>>();
        for candidate in candidates {
            append_orange_candidate(state, output, candidate, center, &loc)?;
        }
        return Ok(Value::List(output));
    }

    let integral_coordinate = |value: f32| -> Option<i32> {
        (value.is_finite()
            && value.fract() == 0.0
            && value >= i32::MIN as f32
            && value <= i32::MAX as f32)
            .then_some(value as i32)
    };
    let (Some(center_x), Some(center_y), Some(center_z)) = (
        integral_coordinate(center_x),
        integral_coordinate(center_y),
        integral_coordinate(center_z),
    ) else {
        return Ok(Value::List(output));
    };
    let distance = distance.min(i32::MAX as f32) as i32;
    let low_x = center_x.saturating_sub(distance);
    let high_x = center_x.saturating_add(distance);
    let low_y = center_y.saturating_sub(distance);
    let high_y = center_y.saturating_add(distance);
    let axis_len = |low: i32, high: i32| {
        u128::try_from(i64::from(high) - i64::from(low) + 1)
            .expect("ordered i32 bounds have a positive span")
    };
    let area = axis_len(low_x, high_x).saturating_mul(axis_len(low_y, high_y));
    let direct_limit = (state.world_turfs.len() as u128)
        .saturating_mul(2)
        .max(4_096);
    let mut tiles = if area <= direct_limit {
        let mut tiles = Vec::new();
        for x in low_x..=high_x {
            for y in low_y..=high_y {
                if let Some(turf) = state.turf_at(x, y, center_z) {
                    tiles.push(((x, y, center_z), turf));
                }
            }
        }
        tiles
    } else {
        state
            .world_turfs
            .iter()
            .filter(|((x, y, z), _)| {
                *z == center_z && *x >= low_x && *x <= high_x && *y >= low_y && *y <= high_y
            })
            .map(|(coordinate, turf)| (*coordinate, *turf))
            .collect::<Vec<_>>()
    };
    let center_coordinate = (center_x, center_y, center_z);
    if let Some(index) = tiles
        .iter()
        .position(|(coordinate, _)| *coordinate == center_coordinate)
    {
        let center_tile = tiles.remove(index);
        tiles.insert(0, center_tile);
    }

    let contents = FieldName::parse("contents").expect("built-in contents field");
    let mut seen_areas = HashSet::new();
    for (coordinate, turf) in tiles {
        append_orange_candidate(state, output, turf, center, &loc)?;
        if let Some(area) = state.world_areas.get(&coordinate).copied()
            && seen_areas.insert(area)
        {
            append_orange_candidate(state, output, area, center, &loc)?;
        }
        let members = state
            .heap
            .datum_field(turf, &contents)
            .ok()
            .and_then(|value| match value {
                Value::List(list) => state.heap.list(*list).ok(),
                _ => None,
            })
            .map(|list| {
                list.positions()
                    .filter_map(|(_, value)| match value {
                        Value::Datum(member) => Some(*member),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for member in members {
            append_orange_candidate(state, output, member, center, &loc)?;
        }
    }
    Ok(Value::List(output))
}

pub(super) fn spatial_query(
    arguments: &[Value],
    state: &mut ExecutionState,
    usr: &Value,
    mobs_only: bool,
    exclude_center: bool,
) -> Result<Value, String> {
    let default_distance = state
        .global(&FieldName::parse("world").expect("built-in world global"))
        .and_then(|world| match world {
            Value::Datum(world) => state
                .heap
                .datum_field(
                    *world,
                    &FieldName::parse("view").expect("built-in world view field"),
                )
                .ok(),
            _ => None,
        })
        .and_then(Value::as_number)
        .filter(|distance| distance.is_finite() && *distance >= 0.0)
        .unwrap_or(5.0)
        .floor();
    let mut distance_x = default_distance;
    let mut distance_y = default_distance;
    let mut center = usr.clone();
    for argument in arguments {
        match argument {
            Value::Null => {}
            Value::Datum(id) => {
                let datum = state.heap.datum(*id).map_err(|error| error.to_string())?;
                let atom = TypePath::parse("/atom").expect("built-in atom path");
                if !is_subtype(state, datum.type_path(), &atom) {
                    return Err(format!(
                        "spatial query center requires an atom, received {argument}"
                    ));
                }
                center = argument.clone();
            }
            Value::Number(value) => {
                distance_x = value.to_f32().floor();
                distance_y = distance_x;
            }
            Value::Text(value) => {
                let (width, height) = value
                    .split_once('x')
                    .or_else(|| value.split_once('X'))
                    .ok_or_else(|| {
                        format!("spatial query distance requires a number or view size, received {argument}")
                    })?;
                let width = width.trim().parse::<u32>().map_err(|_| {
                    format!("spatial query distance has an invalid width: {argument}")
                })?;
                let height = height.trim().parse::<u32>().map_err(|_| {
                    format!("spatial query distance has an invalid height: {argument}")
                })?;
                distance_x = (width / 2) as f32;
                distance_y = (height / 2) as f32;
            }
            _ => {
                return Err(format!(
                    "spatial query requires an atom and optional distance, received {argument}"
                ));
            }
        }
    }
    let output = state.heap.allocate_list();
    let Some((center_x, center_y, center_z)) = datum_coordinates(state, &center) else {
        return Ok(Value::List(output));
    };
    if !distance_x.is_finite() || distance_x < 0.0 || !distance_y.is_finite() || distance_y < 0.0 {
        return Ok(Value::List(output));
    }
    let candidates = if state.world_turfs.is_empty() {
        // Lightweight standalone fixtures may supply coordinate-bearing atoms
        // without constructing canonical world geometry.
        state.heap.datums().map(|(id, _)| id).collect::<Vec<_>>()
    } else {
        indexed_spatial_candidates(
            state,
            center_x,
            center_y,
            center_z,
            distance_x,
            distance_y,
            exclude_center,
        )
    };
    let matching = candidates
        .into_iter()
        .filter_map(|id| {
            let datum = state.heap.datum(id).ok()?;
            let path = datum.type_path().as_str();
            if path == "/area" || path.starts_with("/area/") {
                return None;
            }
            if mobs_only && path != "/mob" && !path.starts_with("/mob/") {
                return None;
            }
            if state.world_turfs.is_empty() {
                let (x, y, z) = datum_coordinates(state, &Value::Datum(id))?;
                if exclude_center && x == center_x && y == center_y && z == center_z {
                    return None;
                }
                return (z == center_z
                    && (x - center_x).abs() <= distance_x
                    && (y - center_y).abs() <= distance_y)
                    .then_some(id);
            }
            // Indexed candidates already came from cells inside the exact
            // requested rectangle and Z level. Center-cell exclusion was
            // applied while walking those cells, before their contents were
            // appended. Avoid resolving loc/x/y/z for every result again.
            Some(id)
        })
        .collect::<Vec<_>>();
    let list = state
        .heap
        .list_mut(output)
        .map_err(|error| error.to_string())?;
    for datum in matching {
        list.add(Value::Datum(datum));
    }
    Ok(Value::List(output))
}

pub(super) fn step_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let Value::Datum(atom) = arguments[0] else {
        return Ok(Value::number(0.0));
    };
    let direction = number(&arguments[1], "step direction")? as i16;
    if direction & !15 != 0 {
        return Ok(Value::number(0.0));
    }
    let Some((x, y, z)) = datum_coordinates(state, &arguments[0]) else {
        return Ok(Value::number(0.0));
    };
    let target = (
        x + f32::from(u8::from(direction & 4 != 0)) - f32::from(u8::from(direction & 8 != 0)),
        y + f32::from(u8::from(direction & 1 != 0)) - f32::from(u8::from(direction & 2 != 0)),
        z,
    );
    // Resolve the destination turf through the shared, world-indexed
    // `get_step` lookup. The previous inline resolver scanned every datum on
    // the heap for each call, so `step`/`step_to`/`step_towards`/`walk_*`
    // scaled with the total atom count. On a fully populated station that is
    // ~0.5 s per call and dominates gib-streak loops during boot.
    // `get_step` retains an all-datums fallback only for geometry-free
    // fixtures, matching the old behaviour there.
    let Value::Datum(turf) = super::get_step_builtin(&arguments[0], &arguments[1], state)? else {
        return Ok(Value::number(0.0));
    };
    let loc_name = FieldName::parse("loc").expect("movement field");
    let old_loc = state
        .heap
        .datum_field(atom, &loc_name)
        .ok()
        .and_then(|value| match value {
            Value::Datum(datum) => Some(*datum),
            _ => None,
        });
    for (name, value) in [
        ("x", Value::number(target.0)),
        ("y", Value::number(target.1)),
        ("z", Value::number(target.2)),
        ("loc", Value::Datum(turf)),
    ] {
        state
            .heap
            .set_datum_field(atom, FieldName::parse(name).expect("movement field"), value)
            .map_err(|error| error.to_string())?;
    }
    if old_loc != Some(turf) {
        synchronize_moved_atom_contents(state, atom, old_loc, Some(turf))?;
    }
    Ok(Value::number(1.0))
}

pub(super) fn direction_between(
    source: &Value,
    target: &Value,
    state: &ExecutionState,
    away: bool,
) -> i16 {
    let (Some((sx, sy, sz)), Some((tx, ty, tz))) = (
        datum_coordinates(state, source),
        datum_coordinates(state, target),
    ) else {
        return 0;
    };
    if sz != tz {
        return 0;
    }
    let (dx, dy) = if away {
        (sx - tx, sy - ty)
    } else {
        (tx - sx, ty - sy)
    };
    let mut direction = 0;
    if dy > 0.0 {
        direction |= 1;
    } else if dy < 0.0 {
        direction |= 2;
    }
    if dx > 0.0 {
        direction |= 4;
    } else if dx < 0.0 {
        direction |= 8;
    }
    direction
}

pub(super) fn step_towards_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let direction = direction_between(&arguments[0], &arguments[1], state, false);
    step_builtin(
        &[arguments[0].clone(), Value::number(f32::from(direction))],
        state,
    )
}

pub(super) fn within_minimum_distance(arguments: &[Value], state: &ExecutionState) -> bool {
    let minimum = arguments.get(2).and_then(Value::as_number).unwrap_or(0.0);
    minimum > 0.0
        && matches!(
            (datum_coordinates(state, &arguments[0]), datum_coordinates(state, &arguments[1])),
            (Some(left), Some(right))
                if left.2 == right.2
                    && (left.0 - right.0).abs().max((left.1 - right.1).abs()) <= minimum
        )
}

pub(super) fn step_to_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if within_minimum_distance(arguments, state) {
        return Ok(Value::number(0.0));
    }
    step_towards_builtin(arguments, state)
}

pub(super) fn get_step_to_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if within_minimum_distance(arguments, state) {
        return Ok(Value::Null);
    }
    let direction = direction_between(&arguments[0], &arguments[1], state, false);
    super::get_step_builtin(&arguments[0], &Value::number(f32::from(direction)), state)
}

pub(super) fn step_away_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if let Some(maximum) = arguments.get(2).and_then(Value::as_number)
        && maximum > 0.0
        && let (Some(source), Some(target)) = (
            datum_coordinates(state, &arguments[0]),
            datum_coordinates(state, &arguments[1]),
        )
        && (source.0 - target.0)
            .abs()
            .max((source.1 - target.1).abs())
            .max((source.2 - target.2).abs())
            > maximum
    {
        return Ok(Value::number(0.0));
    }
    let direction = direction_between(&arguments[0], &arguments[1], state, true);
    step_builtin(
        &[arguments[0].clone(), Value::number(f32::from(direction))],
        state,
    )
}

pub(super) fn get_step_away_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let direction = direction_between(&arguments[0], &arguments[1], state, true);
    super::get_step_builtin(&arguments[0], &Value::number(f32::from(direction)), state)
}

pub(super) fn step_rand_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let directions = [1_i16, 2, 4, 8];
    let index = (super::deterministic_unit(&mut state.random_state) * 4.0) as usize;
    step_builtin(
        &[
            arguments[0].clone(),
            Value::number(f32::from(directions[index.min(3)])),
        ],
        state,
    )
}

pub(super) fn get_step_rand_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let directions = [1_i16, 2, 4, 8];
    let index = (super::deterministic_unit(&mut state.random_state) * 4.0) as usize;
    super::get_step_builtin(
        &arguments[0],
        &Value::number(f32::from(directions[index.min(3)])),
        state,
    )
}

pub(super) fn walk_movable(value: &Value, state: &ExecutionState) -> Option<DatumId> {
    let Value::Datum(datum) = value else {
        return None;
    };
    let path = state.heap().datum(*datum).ok()?.type_path();
    let movable = TypePath::parse("/atom/movable").expect("movable path is valid");
    is_subtype(state, path, &movable).then_some(*datum)
}

pub(super) fn walk_target(value: Option<&Value>, state: &ExecutionState) -> Option<DatumId> {
    let Value::Datum(datum) = value? else {
        return None;
    };
    let path = state.heap().datum(*datum).ok()?.type_path();
    let atom = TypePath::parse("/atom").expect("atom path is valid");
    is_subtype(state, path, &atom).then_some(*datum)
}

pub(super) fn walk_lag(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_number)
        .filter(|lag| lag.is_finite() && *lag > 0.0)
        .map_or(1, |lag| lag.trunc() as u64)
        .max(1)
}

pub(super) fn start_native_walk(
    name: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let Some(movable) = arguments
        .first()
        .and_then(|value| walk_movable(value, state))
    else {
        return Ok(Value::Null);
    };

    let (kind, lag) = match name {
        "walk" => {
            let direction = arguments.get(1).and_then(Value::as_number).unwrap_or(0.0) as i16;
            if direction == 0 {
                state.native_walks.remove(&movable);
                return Ok(Value::Null);
            }
            (
                NativeWalkKind::Direction(direction),
                walk_lag(arguments.get(2)),
            )
        }
        "walk_rand" => (NativeWalkKind::Random, walk_lag(arguments.get(1))),
        "walk_towards" => {
            let Some(target) = walk_target(arguments.get(1), state) else {
                state.native_walks.remove(&movable);
                return Ok(Value::Null);
            };
            (NativeWalkKind::Towards(target), walk_lag(arguments.get(2)))
        }
        "walk_to" => {
            let Some(target) = walk_target(arguments.get(1), state) else {
                state.native_walks.remove(&movable);
                return Ok(Value::Null);
            };
            (
                NativeWalkKind::To {
                    target,
                    minimum: arguments.get(2).and_then(Value::as_number).unwrap_or(0.0),
                },
                walk_lag(arguments.get(3)),
            )
        }
        "walk_away" => {
            let Some(target) = walk_target(arguments.get(1), state) else {
                state.native_walks.remove(&movable);
                return Ok(Value::Null);
            };
            (
                NativeWalkKind::Away {
                    target,
                    maximum: arguments.get(2).and_then(Value::as_number).unwrap_or(5.0),
                },
                walk_lag(arguments.get(3)),
            )
        }
        _ => return Err(format!("unknown native walk procedure {name:?}")),
    };

    let sequence = state.scheduler_sequence;
    state.scheduler_sequence = state.scheduler_sequence.saturating_add(1);
    state.native_walks.insert(
        movable,
        NativeWalk {
            due_tick: state.scheduler_tick.saturating_add(lag),
            sequence,
            lag,
            kind,
        },
    );
    Ok(Value::Null)
}

pub(super) fn native_walk_step(
    movable: DatumId,
    kind: &NativeWalkKind,
    state: &mut ExecutionState,
) -> bool {
    if state.heap().datum(movable).is_err() {
        return false;
    }
    let movable = Value::Datum(movable);
    match *kind {
        NativeWalkKind::Direction(direction) => {
            step_builtin(&[movable, Value::number(f32::from(direction))], state).is_ok()
        }
        NativeWalkKind::Random => step_rand_builtin(&[movable], state).is_ok(),
        NativeWalkKind::Towards(target) => {
            state.heap().datum(target).is_ok()
                && step_towards_builtin(&[movable, Value::Datum(target)], state).is_ok()
        }
        NativeWalkKind::To { target, minimum } => {
            if state.heap().datum(target).is_err() {
                return false;
            }
            let arguments = [movable, Value::Datum(target), Value::number(minimum)];
            if within_minimum_distance(&arguments, state) {
                return false;
            }
            step_to_builtin(&arguments, state).is_ok()
        }
        NativeWalkKind::Away { target, maximum } => {
            if state.heap().datum(target).is_err() {
                return false;
            }
            let arguments = [movable, Value::Datum(target), Value::number(maximum)];
            if maximum > 0.0
                && let (Some(source), Some(target)) = (
                    datum_coordinates(state, &arguments[0]),
                    datum_coordinates(state, &arguments[1]),
                )
                && (source.0 - target.0)
                    .abs()
                    .max((source.1 - target.1).abs())
                    .max((source.2 - target.2).abs())
                    > maximum
            {
                return false;
            }
            step_away_builtin(&arguments, state).is_ok()
        }
    }
}

pub(crate) fn advance_native_walks(state: &mut ExecutionState) {
    let now = state.scheduler_tick;
    let mut due = state
        .native_walks
        .iter()
        .filter(|(_, walk)| walk.due_tick <= now)
        .map(|(movable, walk)| (*movable, walk.due_tick, walk.sequence))
        .collect::<Vec<_>>();
    due.sort_unstable_by_key(|(_, due_tick, sequence)| (*due_tick, *sequence));

    for (movable, _, _) in due {
        let Some(mut walk) = state.native_walks.remove(&movable) else {
            continue;
        };
        let mut active = true;
        while active && walk.due_tick <= now {
            active = native_walk_step(movable, &walk.kind, state);
            walk.due_tick = walk.due_tick.saturating_add(walk.lag);
        }
        if active {
            state.native_walks.insert(movable, walk);
        }
    }
}

pub(super) fn bounds_dist_builtin(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let Some(left) = datum_coordinates(state, &arguments[0]) else {
        return Ok(Value::number(f32::INFINITY));
    };
    let Some(right) = datum_coordinates(state, &arguments[1]) else {
        return Ok(Value::number(f32::INFINITY));
    };
    if left.2 != right.2 {
        return Ok(Value::number(f32::INFINITY));
    }
    let dimension = |value: &Value, name: &str| {
        let Value::Datum(datum) = value else {
            return 32.0;
        };
        let Ok(field) = FieldName::parse(name) else {
            return 32.0;
        };
        super::datum_field_or_initial(state, *datum, &field)
            .ok()
            .as_ref()
            .and_then(Value::as_number)
            .unwrap_or(32.0)
    };
    let horizontal = (right.0 - left.0).abs() * 32.0
        - f32::midpoint(
            dimension(&arguments[0], "bound_width"),
            dimension(&arguments[1], "bound_width"),
        );
    let vertical = (right.1 - left.1).abs() * 32.0
        - f32::midpoint(
            dimension(&arguments[0], "bound_height"),
            dimension(&arguments[1], "bound_height"),
        );
    Ok(Value::number(horizontal.max(vertical)))
}

pub(crate) fn synchronize_moved_atom_contents(
    state: &mut ExecutionState,
    atom: DatumId,
    old_loc: Option<DatumId>,
    new_loc: Option<DatumId>,
) -> Result<(), String> {
    let contents = builtin_contents_field();
    let loc = builtin_loc_field();
    let enclosing_area = |state: &ExecutionState, turf: DatumId| {
        if !state
            .heap
            .datum(turf)
            .is_ok_and(|datum| super::is_turf_type_path(datum.type_path()))
        {
            return None;
        }
        state
            .heap
            .datum_field(turf, loc)
            .ok()
            .and_then(|value| match value {
                Value::Datum(area) => Some(*area),
                _ => None,
            })
    };
    let old_area = old_loc.and_then(|turf| enclosing_area(state, turf));
    let new_area = new_loc.and_then(|turf| enclosing_area(state, turf));
    let contents_list = |state: &ExecutionState, container: DatumId| {
        state
            .heap
            .datum_field(container, contents)
            .ok()
            .and_then(|value| match value {
                Value::List(list) => Some(*list),
                _ => None,
            })
    };
    if let Some(old_loc) = old_loc
        && let Some(list) = contents_list(state, old_loc)
    {
        state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .remove_first(&Value::Datum(atom));
    }
    if let Some(container) = new_loc.filter(|container| *container != atom) {
        let list = state.ensure_contents(container)?;
        state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .add(Value::Datum(atom));
    }
    if old_area != new_area {
        if let Some(list) = old_area.and_then(|area| contents_list(state, area)) {
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_first(&Value::Datum(atom));
        }
        if let Some(list) = new_area.and_then(|area| contents_list(state, area)) {
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .add(Value::Datum(atom));
        }
    }
    Ok(())
}

pub(super) fn get_dist(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    if matches!((&arguments[0], &arguments[1]), (Value::Datum(left), Value::Datum(right)) if left == right)
    {
        return Ok(Value::number(-1.0));
    }
    let Some(left) = datum_coordinates(state, &arguments[0]) else {
        return Ok(Value::number(f32::INFINITY));
    };
    let Some(right) = datum_coordinates(state, &arguments[1]) else {
        return Ok(Value::number(f32::INFINITY));
    };
    Ok(Value::number(
        (left.0 - right.0)
            .abs()
            .max((left.1 - right.1).abs())
            .max((left.2 - right.2).abs()),
    ))
}

pub(crate) fn is_subtype(state: &ExecutionState, candidate: &TypePath, target: &TypePath) -> bool {
    if candidate == target {
        return true;
    }
    if let (Some(candidate), Some(target)) = (
        state.subtype_interval(candidate),
        state.subtype_interval(target),
    ) {
        return target.0 <= candidate.0 && candidate.1 <= target.1;
    }
    let mut current = candidate.clone();
    for _ in 0..512 {
        let parent = if let Some(parent) = state.type_parents.get(&current) {
            parent.clone()
        } else {
            fallback_parent(&current)
        };
        let Some(parent) = parent else {
            return false;
        };
        if &parent == target {
            return true;
        }
        if parent == current {
            return false;
        }
        current = parent;
    }
    false
}

pub(super) fn fallback_parent(path: &TypePath) -> Option<TypePath> {
    let path = path.as_str();
    let explicit = match path {
        "/obj" | "/mob" => Some("/atom/movable"),
        "/area" | "/turf" | "/atom/movable" => Some("/atom"),
        "/atom" => Some("/datum"),
        _ => None,
    };
    if let Some(parent) = explicit {
        return TypePath::parse(parent).ok();
    }
    if let Some(index) = path.rfind('/') {
        if index > 0 {
            return TypePath::parse(&path[..index]).ok();
        }
    }
    TypePath::parse("/datum").ok()
}

pub(super) fn astype(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(target) = arguments.get(1) else {
        // The one-argument form gets its constraint from contextual DM type
        // information; lowering has already validated that context.
        return Ok(arguments[0].clone());
    };
    let Value::TypePath(target) = target else {
        return Ok(Value::Null);
    };
    let candidate = match &arguments[0] {
        Value::Datum(datum) => state
            .heap
            .datum(*datum)
            .map_err(|error| error.to_string())?
            .type_path(),
        Value::TypePath(path) => path,
        _ => return Ok(Value::Null),
    };
    Ok(if is_subtype(state, candidate, target) {
        arguments[0].clone()
    } else {
        Value::Null
    })
}

pub(super) fn turn(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if let Value::Datum(icon) = &arguments[0]
        && super::is_icon_datum(*icon, &state.heap)
    {
        let angle = arguments[1].as_number().unwrap_or(0.0);
        let cloned = super::clone_icon_datum(*icon, &mut state.heap)?;
        super::execute_icon_method(cloned, "Turn", &[Value::number(angle)], &mut state.heap)?;
        return Ok(Value::Datum(cloned));
    }
    if let Value::Datum(matrix) = &arguments[0]
        && super::is_matrix_datum(*matrix, &state.heap)
    {
        let angle = number(&arguments[1], "turn angle")?.to_radians();
        let mut cosine = angle.cos();
        let mut sine = angle.sin();
        if cosine.abs() < 1.0e-6 {
            cosine = 0.0;
        }
        if sine.abs() < 1.0e-6 {
            sine = 0.0;
        }
        let rotated = super::matrix_product(
            super::matrix_components(*matrix, &state.heap)?,
            [cosine, sine, 0.0, -sine, cosine, 0.0],
        );
        return super::allocate_matrix(rotated, &mut state.heap).map(Value::Datum);
    }
    const DIRECTIONS: [i32; 8] = [1, 9, 8, 10, 2, 6, 4, 5];
    let direction = number(&arguments[0], "turn direction")?.trunc() as i32;
    let angle = number(&arguments[1], "turn angle")?;
    let steps = (angle / 45.0).trunc() as i32;
    if steps == 0 {
        return Ok(Value::number(direction as f32));
    }
    let index = DIRECTIONS
        .iter()
        .position(|candidate| *candidate == direction);
    let index = index.unwrap_or_else(|| {
        let sample = super::deterministic_unit(&mut state.random_state);
        (sample * DIRECTIONS.len() as f32).floor() as usize % DIRECTIONS.len()
    });
    let rotated = (index as i32 + steps).rem_euclid(DIRECTIONS.len() as i32) as usize;
    Ok(Value::number(DIRECTIONS[rotated] as f32))
}

pub(super) fn ckey(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let key = runtime_text(&arguments[0], state, "ckey")?;
    Ok(Value::text(
        key.chars()
            .filter(|character| character.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>(),
    ))
}

pub(super) fn ckey_ex(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let key = runtime_text(&arguments[0], state, "ckeyEx")?;
    Ok(Value::text(
        key.chars()
            .filter(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '@' | '_' | '-')
            })
            .collect::<String>(),
    ))
}
