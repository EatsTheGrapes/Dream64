//! Numeric coercion helpers (`number`, `runtime_text`), unary/decrement
//! arithmetic, `arctan`/`log`/`lerp`/`clamp` and extrema.

use dm_value::{FieldName, Value};

use crate::compare_values;

use super::ExecutionState;
pub(super) fn unary_number(
    arguments: &[Value],
    operation: impl FnOnce(f32) -> f32,
) -> Result<Value, String> {
    let value = number(&arguments[0], "numeric builtin")?;
    Ok(Value::number(operation(value)))
}

pub(super) fn extrema_builtin(
    arguments: &[Value],
    state: &ExecutionState,
    maximum: bool,
) -> Result<Value, String> {
    let values = if let [Value::List(list)] = arguments {
        state
            .heap()
            .list(*list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>()
    } else {
        arguments.to_vec()
    };
    let Some(mut result) = values.first().cloned() else {
        return Ok(Value::Null);
    };
    for value in values.iter().skip(1) {
        let ordering = compare_values(value, &result)?;
        if ordering.is_some_and(|ordering| {
            if maximum {
                ordering.is_gt()
            } else {
                ordering.is_lt()
            }
        }) {
            result.clone_from(value);
        }
    }
    Ok(result)
}

fn fallback_number(value: &Value) -> f32 {
    match value {
        Value::Number(number) => number.to_f32(),
        Value::Null
        | Value::Text(_)
        | Value::File(_)
        | Value::TypePath(_)
        | Value::ModifiedTypePath(_)
        | Value::Datum(_)
        | Value::List(_) => 0.0,
    }
}

pub(super) fn inverse_trig(
    arguments: &[Value],
    operation: impl FnOnce(f32) -> f32,
) -> Result<Value, String> {
    let value = fallback_number(&arguments[0]);
    let value = if (-1.0..=1.0).contains(&value) {
        operation(value).to_degrees()
    } else {
        0.0
    };
    Ok(Value::number(value))
}

pub(super) fn arctan_builtin(arguments: &[Value]) -> Result<Value, String> {
    let first = fallback_number(&arguments[0]);
    let value = if arguments.len() == 1 {
        first.atan().to_degrees()
    } else {
        let second = fallback_number(&arguments[1]);
        second.atan2(first).to_degrees()
    };
    Ok(Value::number(value))
}

pub(crate) fn number(value: &Value, context: &str) -> Result<f32, String> {
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        _ => Err(format!("{context} requires a number, received {value}")),
    }
}

pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Number(number) => number.to_f32() != 0.0,
        Value::Text(text) => !text.is_empty(),
        Value::File(_)
        | Value::TypePath(_)
        | Value::ModifiedTypePath(_)
        | Value::Datum(_)
        | Value::List(_) => true,
    }
}

pub(super) fn log_builtin(arguments: &[Value]) -> Result<Value, String> {
    let value = if arguments.len() == 1 {
        number(&arguments[0], "log")?.ln()
    } else {
        let base = number(&arguments[0], "log base")?;
        let value = number(&arguments[1], "log value")?;
        value.log(base)
    };
    Ok(Value::number(value))
}

pub(super) fn lerp_builtin(arguments: &[Value]) -> Result<Value, String> {
    let start = number(&arguments[0], "lerp start")?;
    let end = number(&arguments[1], "lerp end")?;
    let factor = number(&arguments[2], "lerp factor")?;
    Ok(Value::number(start + (end - start) * factor))
}

/// Implements BYOND's scalar and list `clamp(value, low, high)` forms.
/// Bounds are interchangeable. List input produces a new positional list and
/// skips nonnumeric entries, matching Dream Maker's observable behavior.
pub(super) fn clamp_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let mut low = number(&arguments[1], "clamp lower bound")?;
    let mut high = number(&arguments[2], "clamp upper bound")?;
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    if let Value::List(list) = arguments[0] {
        let clamped = state
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::Number(number) => Some(Value::number(number.to_f32().clamp(low, high))),
                _ => None,
            })
            .collect::<Vec<_>>();
        let result = state.heap.allocate_list();
        let list = state
            .heap
            .list_mut(result)
            .map_err(|error| error.to_string())?;
        for value in clamped {
            list.add(value);
        }
        Ok(Value::List(result))
    } else {
        let value = number(&arguments[0], "clamp value")?;
        Ok(Value::number(value.clamp(low, high)))
    }
}

pub(crate) fn runtime_text(
    value: &Value,
    state: &ExecutionState,
    _context: &str,
) -> Result<String, String> {
    match value {
        Value::Text(text) | Value::File(text) => Ok(text.to_string()),
        Value::Null => Ok(String::new()),
        Value::Number(number) => {
            let number = number.to_f32();
            Ok(if number.is_nan() {
                "nan".to_owned()
            } else {
                number.to_string()
            })
        }
        Value::TypePath(path) => Ok(path.to_string()),
        Value::ModifiedTypePath(path) => Ok(path.base().to_string()),
        Value::Datum(datum) => {
            let datum = state
                .heap
                .datum(*datum)
                .map_err(|error| error.to_string())?;
            let name = FieldName::parse("name").expect("built-in datum name is valid");
            if let Ok(Value::Text(name)) = datum.field(&name) {
                return Ok(name.to_string());
            }
            Ok(datum.type_path().to_string())
        }
        // BYOND exposes lists as the engine datum display name rather than
        // joining their contents. Verified on 516.1680 for both positional
        // and associative lists: `"[L]"` is exactly `/list`.
        Value::List(_) => Ok("/list".to_owned()),
    }
}
