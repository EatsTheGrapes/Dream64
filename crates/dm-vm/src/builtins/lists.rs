//! List mutation procedures (`list`, `values`, and infix list operator
//! support): `list_add`, `list_copy`, `list_cut`, `list_find`,
//! `list_insert`, `list_join`, `list_remove`, `list_splice`, `list_swap`,
//! and the `operator+/-/*/@/?` entry snapshot bookkeeping shared with the
//! runtime heap.

use dm_value::{DatumId, ListId, Value};

use smallvec::SmallVec;

use super::{
    CompoundAssignmentOperator, ExecutionState, builtin_contents_field, builtin_coordinate_fields,
    builtin_loc_field, runtime_text, synchronize_moved_atom_contents,
};

#[derive(Clone)]
pub(super) struct ListOperatorEntry {
    pub(super) key: Value,
    pub(super) associated: Option<Value>,
}

pub(super) fn list_operator_snapshot(
    list: ListId,
    state: &ExecutionState,
) -> Result<Vec<ListOperatorEntry>, String> {
    let list = state.heap.list(list).map_err(|error| error.to_string())?;
    Ok(list
        .positions()
        // A destroyed engine object can remain as an invalid arena handle in
        // a long-lived bookkeeping list until its owning DM subsystem removes
        // it. Never propagate that handle into a newly constructed list: it
        // would canonicalize to null only when the consumer iterates it. This
        // distinction preserves an explicitly stored DM null while keeping
        // list union/addition results free of dead object references.
        .filter(|(_, key)| match key {
            Value::Datum(datum) => state.heap.datum(*datum).is_ok(),
            Value::List(nested) => state.heap.list(*nested).is_ok(),
            _ => true,
        })
        .map(|(_, key)| {
            let associated = list.get_key(key).ok().cloned();
            ListOperatorEntry {
                key: key.clone(),
                associated,
            }
        })
        .collect())
}

pub(super) fn add_operator_entry(
    list: ListId,
    entry: ListOperatorEntry,
    state: &mut ExecutionState,
    only_if_absent: bool,
) -> Result<(), String> {
    let preserve_existing_key = state.is_associative_list(list);
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    if (only_if_absent || preserve_existing_key) && target.contains(&entry.key) {
        return Ok(());
    }
    if let Some(associated) = entry.associated {
        target.set_key(entry.key, associated);
    } else {
        target.add(entry.key);
    }
    Ok(())
}

pub(super) fn remove_all_operator_matches(
    list: ListId,
    value: &Value,
    state: &mut ExecutionState,
) -> Result<usize, String> {
    let mut removed = 0;
    while state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .remove_last(value)
        .is_some()
    {
        removed += 1;
    }
    Ok(removed)
}

pub(super) fn operator_rhs_entries(
    value: &Value,
    state: &ExecutionState,
) -> Result<Vec<ListOperatorEntry>, String> {
    if let Value::List(list) = value {
        list_operator_snapshot(*list, state)
    } else {
        Ok(vec![ListOperatorEntry {
            key: value.clone(),
            associated: None,
        }])
    }
}

pub(crate) fn execute_list_binary_operator(
    operator: &str,
    left: ListId,
    right: &Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match operator {
        "+" => {
            let result = state
                .heap
                .copy_list(left)
                .map_err(|error| error.to_string())?;
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(result, entry, state, false)?;
            }
            Ok(Value::List(result))
        }
        "-" => {
            let result = state
                .heap
                .copy_list(left)
                .map_err(|error| error.to_string())?;
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            let keys = operator_rhs_entries(right, state)?
                .into_iter()
                .map(|entry| entry.key)
                .collect::<Vec<_>>();
            state
                .heap
                .list_mut(result)
                .map_err(|error| error.to_string())?
                .subtract_entries(&keys)
                .map_err(|error| error.to_string())?;
            Ok(Value::List(result))
        }
        "|" => {
            let result = state.heap.allocate_list();
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            for entry in list_operator_snapshot(left, state)? {
                add_operator_entry(result, entry, state, true)?;
            }
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(result, entry, state, true)?;
            }
            Ok(Value::List(result))
        }
        "&" => {
            let result = state
                .heap
                .copy_list(left)
                .map_err(|error| error.to_string())?;
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            let right_entries = operator_rhs_entries(right, state)?;
            let snapshot = list_operator_snapshot(result, state)?;
            for entry in snapshot {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(result, &entry.key, state)?;
                }
            }
            Ok(Value::List(result))
        }
        "^" => {
            let result = state.heap.allocate_list();
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            let left_entries = list_operator_snapshot(left, state)?;
            let right_entries = operator_rhs_entries(right, state)?;
            for entry in &left_entries {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    add_operator_entry(result, entry.clone(), state, true)?;
                }
            }
            for entry in right_entries {
                if !left_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    add_operator_entry(result, entry, state, true)?;
                }
            }
            Ok(Value::List(result))
        }
        _ => Err(format!("unsupported /list binary operator {operator:?}")),
    }
}

pub(crate) fn execute_list_compound_operator(
    operator: CompoundAssignmentOperator,
    left: ListId,
    right: &Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if !matches!(right, Value::List(_)) {
        let incremental = match operator {
            CompoundAssignmentOperator::Add => {
                state.mutate_vis_contents_scalar(left, right, true)?
            }
            CompoundAssignmentOperator::Subtract => {
                state.mutate_vis_contents_scalar(left, right, false)?
            }
            _ => None,
        };
        if incremental.is_some() {
            return Ok(Value::List(left));
        }
    }
    let visibility_before = state
        .is_visibility_list(left)
        .then(|| state.visibility_members(left))
        .transpose()?;
    match operator {
        CompoundAssignmentOperator::Add => {
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(left, entry, state, false)?;
            }
        }
        CompoundAssignmentOperator::Subtract => {
            let keys = operator_rhs_entries(right, state)?
                .into_iter()
                .map(|entry| entry.key)
                .collect::<Vec<_>>();
            state
                .heap
                .list_mut(left)
                .map_err(|error| error.to_string())?
                .subtract_entries(&keys)
                .map_err(|error| error.to_string())?;
        }
        CompoundAssignmentOperator::BitOr => {
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(left, entry, state, true)?;
            }
        }
        CompoundAssignmentOperator::BitAnd => {
            let right_entries = operator_rhs_entries(right, state)?;
            let snapshot = list_operator_snapshot(left, state)?;
            for entry in snapshot {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(left, &entry.key, state)?;
                }
            }
        }
        CompoundAssignmentOperator::BitXor => {
            let right_entries = operator_rhs_entries(right, state)?;
            let original = list_operator_snapshot(left, state)?;
            for entry in right_entries {
                if original
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(left, &entry.key, state)?;
                } else {
                    add_operator_entry(left, entry, state, true)?;
                }
            }
        }
        CompoundAssignmentOperator::Multiply
        | CompoundAssignmentOperator::Divide
        | CompoundAssignmentOperator::Remainder
        | CompoundAssignmentOperator::FractionalRemainder
        | CompoundAssignmentOperator::ShiftLeft
        | CompoundAssignmentOperator::ShiftRight => {
            return Err(format!(
                "operator {operator:?} is not defined for a BYOND list"
            ));
        }
    }
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(left, &before)?;
    }
    Ok(Value::List(left))
}

pub(crate) fn execute_list_method(
    name: &str,
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Option<Result<Value, String>> {
    let alist = state.is_associative_list(list);
    Some(match name {
        "Add" => list_add(list, arguments, state),
        "Copy" if alist && !arguments.is_empty() => {
            Err("alist.Copy does not accept range arguments".to_owned())
        }
        "Copy" => list_copy(list, arguments, state),
        "Cut" => list_cut(list, arguments, state),
        "Find" => list_find(list, arguments, state),
        "Insert" if alist => Err("alist.Insert is not supported".to_owned()),
        "Insert" => list_insert(list, arguments, state),
        "Join" => list_join(list, arguments, state),
        "Remove" => list_remove(list, arguments, state, false),
        "RemoveAll" => list_remove(list, arguments, state, true),
        "Splice" if alist => Err("alist.Splice is not supported".to_owned()),
        "Splice" => list_splice(list, arguments, state),
        "Swap" if alist => Err("alist.Swap is not supported".to_owned()),
        "Swap" => list_swap(list, arguments, state),
        _ => return None,
    })
}

pub(super) fn list_integer(
    value: Option<&Value>,
    default: i64,
    context: &str,
) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) if number.to_f32().is_finite() => {
            Ok(number.to_f32().trunc() as i64)
        }
        Some(value) => Err(format!(
            "{context} requires a numeric index, received {value}"
        )),
    }
}

pub(super) fn list_boundary(value: i64, len: usize, zero_is_end: bool) -> Result<usize, String> {
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let value = if value == 0 {
        if zero_is_end { limit } else { 1 }
    } else {
        value
    };
    if value < 1 || value > limit {
        return Err(format!("list index {value} is outside 1 through {limit}"));
    }
    usize::try_from(value).map_err(|error| format!("list index is not representable: {error}"))
}

pub(super) fn splice_boundary(value: i64, len: usize, zero_is_end: bool) -> usize {
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let value = if value == 0 && zero_is_end {
        limit
    } else if value < 0 {
        limit.saturating_add(value)
    } else {
        value
    };
    usize::try_from(value.clamp(1, limit)).unwrap_or(usize::MAX)
}

pub(super) fn flattened_list_arguments(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Vec<Value>, String> {
    let mut values = Vec::new();
    for argument in arguments {
        if let Value::List(list) = argument {
            let snapshot = state
                .heap
                .list(*list)
                .map_err(|error| error.to_string())?
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>();
            values.extend(snapshot);
        } else {
            values.push(argument.clone());
        }
    }
    Ok(values)
}

pub(super) fn list_add(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.is_empty() {
        return Err("list.Add requires at least one item".to_owned());
    }
    if let [value] = arguments
        && !matches!(value, Value::List(_))
        && state
            .mutate_vis_contents_scalar(list, value, true)?
            .is_some()
    {
        return Ok(Value::Null);
    }
    let values = flattened_list_arguments(arguments, state)?;
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    if let Some(owner) = state.contents_owner(list) {
        let owner_path = state
            .heap
            .datum(owner)
            .map_err(|error| error.to_string())?
            .type_path()
            .as_str()
            .to_owned();
        if owner_path == "/area" || owner_path.starts_with("/area/") {
            for value in values {
                let Value::Datum(turf) = value else {
                    return Err("area.contents.Add requires a turf".to_owned());
                };
                let path = state
                    .heap
                    .datum(turf)
                    .map_err(|error| error.to_string())?
                    .type_path()
                    .as_str();
                if path != "/turf" && !path.starts_with("/turf/") {
                    return Err("area.contents.Add requires a turf".to_owned());
                }
                move_turf_to_area(state, turf, owner)?;
            }
            return Ok(Value::Null);
        }
        if owner_path == "/turf" || owner_path.starts_with("/turf/") {
            for value in values {
                let Value::Datum(movable) = value else {
                    return Err("turf.contents.Add requires a movable atom".to_owned());
                };
                let path = state
                    .heap
                    .datum(movable)
                    .map_err(|error| error.to_string())?
                    .type_path()
                    .as_str();
                if !is_movable_path(path) {
                    return Err("turf.contents.Add requires a movable atom".to_owned());
                }
                move_movable_to_turf(state, movable, owner)?;
            }
            return Ok(Value::Null);
        }
    }
    let associative_only = state.is_associative_list(list);
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for value in values {
        if associative_only {
            if !target.contains(&value) {
                target.set_key(value, Value::Null);
            }
        } else {
            target.add(value);
        }
    }
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(Value::Null)
}

pub(crate) fn is_movable_path(path: &str) -> bool {
    path == "/obj"
        || path.starts_with("/obj/")
        || path == "/mob"
        || path.starts_with("/mob/")
        || path == "/atom/movable"
        || path.starts_with("/atom/movable/")
}

pub(crate) fn move_movable_to_turf(
    state: &mut ExecutionState,
    movable: DatumId,
    turf: DatumId,
) -> Result<(), String> {
    let loc = builtin_loc_field();
    let old_loc = state
        .heap
        .datum_field(movable, loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(datum) => Some(*datum),
            _ => None,
        });
    let coordinates = builtin_coordinate_fields();
    let values = coordinates
        .iter()
        .map(|field| {
            state
                .heap
                .datum_field(turf, field)
                .cloned()
                .unwrap_or(Value::Null)
        })
        .collect::<Vec<_>>();
    state
        .heap
        .set_datum_field(movable, loc.clone(), Value::Datum(turf))
        .map_err(|error| error.to_string())?;
    for (field, value) in coordinates.iter().cloned().zip(values) {
        state
            .heap
            .set_datum_field(movable, field, value)
            .map_err(|error| error.to_string())?;
    }
    if old_loc != Some(turf) {
        synchronize_moved_atom_contents(state, movable, old_loc, Some(turf))?;
    }
    Ok(())
}

pub(crate) fn move_movable_to_atom(
    state: &mut ExecutionState,
    movable: DatumId,
    location: DatumId,
) -> Result<(), String> {
    // BYOND does not permit a movable to contain itself, directly or through
    // one of its descendants. Besides corrupting contents, such a cycle makes
    // recursive movement notifications recurse forever.
    let loc = builtin_loc_field();
    let mut cursor = Some(location);
    let mut visited = SmallVec::<[DatumId; 8]>::new();
    while let Some(container) = cursor {
        if container == movable || visited.contains(&container) {
            return Ok(());
        }
        visited.push(container);
        cursor = state
            .heap
            .datum_field(container, loc)
            .ok()
            .and_then(|value| match value {
                Value::Datum(parent) => Some(*parent),
                _ => None,
            });
    }

    let location_is_turf = state.heap.datum(location).is_ok_and(|datum| {
        let path = datum.type_path().as_str();
        path == "/turf" || path.starts_with("/turf/")
    });
    if location_is_turf {
        return move_movable_to_turf(state, movable, location);
    }

    let old_loc = state
        .heap
        .datum_field(movable, loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(datum) => Some(*datum),
            _ => None,
        });
    state
        .heap
        .set_datum_field(movable, loc.clone(), Value::Datum(location))
        .map_err(|error| error.to_string())?;
    if old_loc != Some(location) {
        synchronize_moved_atom_contents(state, movable, old_loc, Some(location))?;
    }
    Ok(())
}

pub(crate) fn move_turf_to_area(
    state: &mut ExecutionState,
    turf: DatumId,
    new_area: DatumId,
) -> Result<(), String> {
    let loc = builtin_loc_field();
    let contents = builtin_contents_field();
    let old_area = state
        .heap
        .datum_field(turf, loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(area) => Some(*area),
            _ => None,
        });
    if old_area == Some(new_area) {
        return Ok(());
    }
    let contained = state
        .heap
        .datum_field(turf, contents)
        .ok()
        .and_then(|value| match value {
            Value::List(list) => state.heap.list(*list).ok(),
            _ => None,
        })
        .map(|list| {
            list.positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if let Some(old_area) = old_area
        && let Ok(Value::List(list)) = state.heap.datum_field(old_area, contents)
    {
        let list = *list;
        let values = std::iter::once(Value::Datum(turf)).chain(contained.iter().cloned());
        let target = state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?;
        for value in values {
            target.remove_first(&value);
        }
    }
    let new_contents = state.ensure_contents(new_area)?;
    {
        let target = state
            .heap
            .list_mut(new_contents)
            .map_err(|error| error.to_string())?;
        for value in std::iter::once(Value::Datum(turf)).chain(contained) {
            if !target.contains(&value) {
                target.add(value);
            }
        }
    }
    state
        .heap
        .set_datum_field(turf, loc.clone(), Value::Datum(new_area))
        .map_err(|error| error.to_string())?;
    state.note_turf_area(turf, new_area);
    Ok(())
}

pub(super) fn list_copy(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() > 2 {
        return Err("list.Copy accepts Start and End only".to_owned());
    }
    state.refresh_vars_proxy(list)?;
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let start = list_boundary(
        list_integer(arguments.first(), 1, "list.Copy Start")?,
        len,
        false,
    )?;
    let end = list_boundary(
        list_integer(arguments.get(1), 0, "list.Copy End")?,
        len,
        true,
    )?;
    let copy = source
        .copy_range(start, end)
        .map_err(|error| error.to_string())?;
    let result = state.heap.allocate_list();
    *state
        .heap
        .list_mut(result)
        .map_err(|error| error.to_string())? = copy;
    if state.is_associative_list(list) {
        state.mark_associative_list(result);
    }
    Ok(Value::List(result))
}

pub(super) fn list_cut(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() > 2 {
        return Err("list.Cut accepts Start and End only".to_owned());
    }
    let len = state
        .heap
        .list(list)
        .map_err(|error| error.to_string())?
        .len();
    let raw_start = list_integer(arguments.first(), 1, "list.Cut Start")?;
    if raw_start < 0 {
        return Err("list.Cut Start cannot be negative".to_owned());
    }
    let start = list_boundary(
        raw_start.min(i64::try_from(len + 1).unwrap_or(i64::MAX)),
        len,
        false,
    )?;
    let raw_end = list_integer(arguments.get(1), 0, "list.Cut End")?;
    if raw_end < 0 {
        return Err("list.Cut End cannot be negative".to_owned());
    }
    let end = if raw_end == 0 || raw_end > i64::try_from(len + 1).unwrap_or(i64::MAX) {
        len + 1
    } else {
        list_boundary(raw_end, len, true)?
    };
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .cut_range(start, end)
        .map_err(|error| error.to_string())?;
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(Value::Null)
}

pub(super) fn list_find(
    list: ListId,
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    if arguments.is_empty() || arguments.len() > 3 {
        return Err("list.Find requires Elem and optional Start/End".to_owned());
    }
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let raw_start = list_integer(arguments.get(1), 1, "list.Find Start")
        .unwrap_or(1)
        .max(1);
    let start = usize::try_from(raw_start)
        .unwrap_or(usize::MAX)
        .min(len.saturating_add(1));
    let raw_end = list_integer(arguments.get(2), 0, "list.Find End").unwrap_or(0);
    let end = if raw_end <= 0 || raw_end > i64::try_from(len + 1).unwrap_or(i64::MAX) {
        len + 1
    } else {
        usize::try_from(raw_end).unwrap_or(len + 1)
    };
    let found = source
        .find_position(&arguments[0], start.max(1), end.max(1))
        .map_err(|error| error.to_string())?;
    Ok(Value::number(found as f32))
}

pub(super) fn list_insert(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() < 2 {
        return Err("list.Insert requires Index and at least one item".to_owned());
    }
    let len = state
        .heap
        .list(list)
        .map_err(|error| error.to_string())?
        .len();
    let raw = list_integer(arguments.first(), 0, "list.Insert Index")?;
    let mut index = if raw <= 0 {
        len + 1
    } else {
        usize::try_from(raw).map_err(|error| format!("list.Insert index is invalid: {error}"))?
    };
    if index > len + 1 {
        return Err(format!("list.Insert index {index} exceeds {}", len + 1));
    }
    let values = flattened_list_arguments(&arguments[1..], state)?;
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for value in values {
        target
            .insert(index, value)
            .map_err(|error| error.to_string())?;
        index += 1;
    }
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(Value::number(index as f32))
}

pub(super) fn list_join(
    list: ListId,
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    if arguments.len() > 3 {
        return Err("list.Join accepts optional Glue, Start, and End".to_owned());
    }
    // BYOND declares Glue as a string but permits it to be omitted. OpenDream
    // observes the missing slot as null and TryGetValueAsString consequently
    // supplies an empty separator. Monkestation relies on this exact shape in
    // `generate_icon_key().Join()` while building human preview appearances.
    let glue = arguments.first().map_or_else(
        || Ok(String::new()),
        |value| runtime_text(value, state, "list.Join Glue"),
    )?;
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let mut start = list_integer(arguments.get(1), 1, "list.Join Start").unwrap_or(1);
    let mut end = list_integer(arguments.get(2), 0, "list.Join End").unwrap_or(0);
    if end <= 0 {
        end = end.saturating_add(limit);
    }
    if start < 0 {
        start = start.saturating_add(limit);
    }
    if start == 0 || start >= end {
        return Ok(Value::text(""));
    }
    let start = usize::try_from(start.max(1))
        .unwrap_or(usize::MAX)
        .min(len + 1);
    let end = usize::try_from(end.max(1))
        .unwrap_or(usize::MAX)
        .min(len + 1);
    let mut values = Vec::new();
    for index in start..end {
        values.push(runtime_text(
            source.get(index).map_err(|error| error.to_string())?,
            state,
            "list.Join item",
        )?);
    }
    Ok(Value::text(values.join(&glue)))
}

pub(super) fn list_remove_once(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<usize, String> {
    let mut removed = 0usize;
    for argument in arguments {
        if matches!(argument, Value::List(candidate) if *candidate == list) {
            let len = state
                .heap
                .list(list)
                .map_err(|error| error.to_string())?
                .len();
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .resize(0)
                .map_err(|error| error.to_string())?;
            removed += len;
            break;
        }
        let values = flattened_list_arguments(std::slice::from_ref(argument), state)?;
        for value in values {
            if state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_last(&value)
                .is_some()
            {
                removed += 1;
            }
        }
    }
    Ok(removed)
}

pub(super) fn list_remove(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
    all: bool,
) -> Result<Value, String> {
    if arguments.is_empty() {
        return Err(if all {
            "list.RemoveAll requires at least one item"
        } else {
            "list.Remove requires at least one item"
        }
        .to_owned());
    }
    if let [value] = arguments
        && !matches!(value, Value::List(_))
        && let Some(removed) = state.mutate_vis_contents_scalar(list, value, false)?
    {
        return Ok(Value::number(f32::from(removed)));
    }
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    let result = if all {
        let mut total = 0usize;
        loop {
            let removed = list_remove_once(list, arguments, state)?;
            total += removed;
            if removed == 0 {
                break;
            }
        }
        Value::number(total as f32)
    } else {
        Value::number(f32::from(list_remove_once(list, arguments, state)? > 0))
    };
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(result)
}

pub(super) fn list_splice(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() > 2 && arguments.len() < 3 {
        return Err("invalid list.Splice arguments".to_owned());
    }
    let len = state
        .heap
        .list(list)
        .map_err(|error| error.to_string())?
        .len();
    let mut start = splice_boundary(
        list_integer(arguments.first(), 1, "list.Splice Start")?,
        len,
        false,
    );
    let mut end = splice_boundary(
        list_integer(arguments.get(1), 0, "list.Splice End")?,
        len,
        true,
    );
    if end < start {
        std::mem::swap(&mut start, &mut end);
    }
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .cut_range(start, end)
        .map_err(|error| error.to_string())?;
    if arguments.len() <= 2 {
        if let Some(before) = visibility_before {
            state.normalize_and_synchronize_visibility_list(list, &before)?;
        }
        return Ok(Value::Null);
    }
    let values = flattened_list_arguments(&arguments[2..], state)?;
    let index = start.min(
        state
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .len()
            + 1,
    );
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for (offset, value) in values.into_iter().enumerate() {
        target
            .insert(index + offset, value)
            .map_err(|error| error.to_string())?;
    }
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(Value::Null)
}

pub(super) fn list_swap(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() != 2 {
        return Err("list.Swap requires exactly two indices".to_owned());
    }
    let first = list_integer(arguments.first(), 0, "list.Swap Index1")?;
    let second = list_integer(arguments.get(1), 0, "list.Swap Index2")?;
    let first = usize::try_from(first).map_err(|_| "list.Swap Index1 is invalid".to_owned())?;
    let second = usize::try_from(second).map_err(|_| "list.Swap Index2 is invalid".to_owned())?;
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .swap(first, second)
        .map_err(|error| error.to_string())?;
    Ok(Value::Null)
}
