//! Parameter-list splice helpers and the `values_cut`/`values_dot`/`values_fold`
//! lattice operations shared by the dispatcher and `params2list`.

use dm_value::Value;

use super::{ExecutionState, lists::list_operator_snapshot, number, truthy};
pub(super) fn values_cut(
    arguments: &[Value],
    state: &mut ExecutionState,
    over: bool,
) -> Result<Value, String> {
    let Value::List(list) = arguments[0] else {
        return Ok(Value::number(0.0));
    };
    let threshold = number(&arguments[1], "values_cut threshold")?;
    let inclusive = arguments.get(2).is_some_and(truthy);
    let snapshot = list_operator_snapshot(list, state)?;
    let mut removed = 0_usize;
    for entry in snapshot {
        let remove = entry
            .associated
            .as_ref()
            .and_then(Value::as_number)
            .is_none_or(|value| {
                if over {
                    value > threshold || (inclusive && value == threshold)
                } else {
                    value < threshold || (inclusive && value == threshold)
                }
            });
        if remove
            && state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_key(&entry.key)
                .or_else(|| state.heap.list_mut(list).ok()?.remove_last(&entry.key))
                .is_some()
        {
            removed += 1;
        }
    }
    Ok(Value::number(removed as f32))
}

pub(super) fn values_dot(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let (Value::List(left), Value::List(right)) = (&arguments[0], &arguments[1]) else {
        return Ok(Value::number(0.0));
    };
    let left = state.heap.list(*left).map_err(|error| error.to_string())?;
    let right = state.heap.list(*right).map_err(|error| error.to_string())?;
    let total = left.positions().fold(0.0, |total, (_, key)| {
        let Some(left_value) = left.get_key(key).ok().and_then(Value::as_number) else {
            return total;
        };
        let Some(right_value) = right.get_key(key).ok().and_then(Value::as_number) else {
            return total;
        };
        total + left_value * right_value
    });
    Ok(Value::number(total))
}

pub(super) fn values_fold(
    arguments: &[Value],
    state: &ExecutionState,
    product: bool,
) -> Result<Value, String> {
    let Value::List(list) = arguments[0] else {
        return Ok(Value::number(0.0));
    };
    let list = state.heap.list(list).map_err(|error| error.to_string())?;
    let mut values = list
        .positions()
        .filter_map(|(_, key)| list.get_key(key).ok().and_then(Value::as_number));
    let result = if product {
        values
            .next()
            .map_or(0.0, |first| values.fold(first, |a, b| a * b))
    } else {
        values.sum()
    };
    Ok(Value::number(result))
}
