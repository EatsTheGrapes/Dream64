//! `json_encode`/`json_decode` plus the heartbeat bridge envelopes that
//! serialize native DM values.

use dm_value::Value;

use super::{ExecutionState, runtime_text};
pub(super) fn json_encode_builtin(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let pretty = arguments
        .get(1)
        .and_then(Value::as_number)
        .is_some_and(|flags| flags.trunc() as i32 & 1 != 0);
    let value = arguments.first().unwrap_or(&Value::Null);
    let json = json_value_from_dm(value, state, 0)?;
    let encoded = if pretty {
        serde_json::to_string_pretty(&json)
    } else {
        serde_json::to_string(&json)
    }
    .map_err(|error| format!("json_encode failed: {error}"))?;
    Ok(Value::text(encoded))
}

fn json_value_from_dm(
    value: &Value,
    state: &ExecutionState,
    depth: usize,
) -> Result<serde_json::Value, String> {
    if depth >= 20 {
        return Ok(serde_json::Value::Null);
    }
    match value {
        Value::Null => Ok(serde_json::Value::Null),
        Value::Number(number) => {
            let number = number.to_f32();
            if number.is_finite() {
                let json_number =
                    number
                        .to_string()
                        .parse::<serde_json::Number>()
                        .map_err(|error| {
                            format!("json_encode cannot encode number {number}: {error}")
                        })?;
                Ok(serde_json::Value::Number(json_number))
            } else {
                let spelling = if number.is_nan() {
                    "NaN"
                } else if number.is_sign_positive() {
                    "Infinity"
                } else {
                    "-Infinity"
                };
                let mut object = serde_json::Map::new();
                object.insert(
                    "__number__".to_owned(),
                    serde_json::Value::String(spelling.to_owned()),
                );
                Ok(serde_json::Value::Object(object))
            }
        }
        Value::Text(text) | Value::File(text) => Ok(serde_json::Value::String(text.to_string())),
        Value::TypePath(path) => Ok(serde_json::Value::String(path.to_string())),
        Value::ModifiedTypePath(path) => Ok(serde_json::Value::String(path.base().to_string())),
        Value::Datum(_) => Ok(serde_json::Value::String(runtime_text(
            value,
            state,
            "json_encode datum",
        )?)),
        Value::List(id) => {
            let list = state.heap.list(*id).map_err(|error| error.to_string())?;
            let entries = list
                .positions()
                .map(|(_, key)| Ok((key.clone(), list.get_key(key).ok().cloned())))
                .collect::<Result<Vec<_>, String>>()?;
            if list.associative_len() == 0 {
                entries
                    .into_iter()
                    .map(|(value, _)| json_value_from_dm(&value, state, depth + 1))
                    .collect::<Result<Vec<_>, _>>()
                    .map(serde_json::Value::Array)
            } else {
                let mut object = serde_json::Map::new();
                for (key, associated) in entries {
                    let key = runtime_text(&key, state, "json_encode list key")?;
                    let value = associated.map_or(Ok(serde_json::Value::Null), |value| {
                        json_value_from_dm(&value, state, depth + 1)
                    })?;
                    object.insert(key, value);
                }
                Ok(serde_json::Value::Object(object))
            }
        }
    }
}

pub(super) fn json_decode_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let Some(Value::Text(text)) = arguments.first() else {
        return Err("json_decode requires text".to_owned());
    };
    let json: serde_json::Value = serde_json::from_str(text).map_err(|error| {
        let preview = text.chars().take(256).collect::<String>();
        format!("json_decode failed for {preview:?}: {error}")
    })?;
    dm_value_from_json(&json, state)
}

fn dm_value_from_json(
    json: &serde_json::Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match json {
        serde_json::Value::Null => Ok(Value::Null),
        serde_json::Value::Bool(value) => Ok(Value::number(f32::from(*value))),
        serde_json::Value::Number(value) => {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("json_decode invalid number {value}"))?
                as f32;
            if !number.is_finite() {
                return Err(format!("json_decode number is outside DM's range: {value}"));
            }
            Ok(Value::number(number))
        }
        serde_json::Value::String(value) => Ok(Value::text(value.clone())),
        serde_json::Value::Array(values) => {
            let decoded = values
                .iter()
                .map(|value| dm_value_from_json(value, state))
                .collect::<Result<Vec<_>, _>>()?;
            let id = state.heap.allocate_list();
            let list = state.heap.list_mut(id).map_err(|error| error.to_string())?;
            for value in decoded {
                list.add(value);
            }
            Ok(Value::List(id))
        }
        serde_json::Value::Object(object) => {
            if object.len() == 1 {
                if let Some(serde_json::Value::String(number)) = object.get("__number__") {
                    let value = match number.as_str() {
                        "NaN" => f32::NAN,
                        "Infinity" => f32::INFINITY,
                        "-Infinity" => f32::NEG_INFINITY,
                        _ => number.parse::<f32>().map_err(|_| {
                            format!("json_decode invalid special number {number:?}")
                        })?,
                    };
                    return Ok(Value::number(value));
                }
            }
            let decoded = object
                .iter()
                .map(|(key, value)| {
                    dm_value_from_json(value, state).map(|value| (Value::text(key.clone()), value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let id = state.heap.allocate_list();
            let list = state.heap.list_mut(id).map_err(|error| error.to_string())?;
            for (key, value) in decoded {
                list.set_key(key, value);
            }
            Ok(Value::List(id))
        }
    }
}
