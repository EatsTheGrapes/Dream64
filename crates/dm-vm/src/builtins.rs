//! Native implementations of documented BYOND global procedures.
//!
//! These routines are deliberately runtime primitives rather than injected DM
//! source when their behavior depends on host state, type metadata, or precise
//! text-indexing semantics.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::unnecessary_wraps,
    reason = "DM uses binary32 numbers for integer/index boundaries and native builtin dispatch shares a Result ABI"
)]

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use dm_value::{FieldName, ListId, TypePath, Value};

use super::{CompoundAssignmentOperator, ExecutionState};

pub(super) fn standard_builtin_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "abs" | "ceil" | "floor" | "fract" | "trunc" | "sign" | "sqrt" | "sin" | "cos" | "tan"
        | "arcsin" | "arccos" | "length_char" | "lowertext" | "uppertext" | "trimtext"
        | "ascii2text" | "text2path" | "isinf" | "isnan" | "ckey" | "fexists" | "file2text"
        | "lentext" | "list2params" | "params2list" => (1, 1),
        "json_decode" | "md5" => (0, 1),
        "json_encode" => (0, 2),
        "log" | "arctan" | "text2ascii" | "text2ascii_char" | "text2num" => (1, 2),
        "image" | "sort_list" | "qdel" | "typecacheof" => (0, 5),
        "clamp" | "lerp" => (3, 3),
        "cmptext" | "cmptextEx" | "sorttext" | "sorttextEx" | "sortText" | "addtext" => {
            (0, usize::MAX)
        }
        "num2text" => (1, 3),
        "findtext"
        | "findtextEx"
        | "findtext_char"
        | "findtextEx_char"
        | "findlasttext"
        | "findlasttextEx"
        | "findlasttext_char"
        | "findlasttextEx_char"
        | "jointext" => (2, 4),
        "splittext" | "splittext_char" => (2, 5),
        "spantext" | "spantext_char" | "nonspantext" | "nonspantext_char" => (2, 3),
        "splicetext" | "splicetext_char" => (4, 4),
        "get_dist" | "turn" | "astype" => (2, 2),
        _ => return None,
    })
}

pub(super) fn execute_standard_builtin(
    name: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match name {
        "qdel" => qdel_builtin(arguments, state),
        "abs" => unary_number(arguments, f32::abs),
        "ceil" => unary_number(arguments, f32::ceil),
        "floor" => unary_number(arguments, f32::floor),
        "fract" => unary_number(arguments, f32::fract),
        "trunc" => unary_number(arguments, f32::trunc),
        "sign" => unary_number(arguments, |value| {
            if value > 0.0 {
                1.0
            } else if value < 0.0 {
                -1.0
            } else {
                value
            }
        }),
        "sqrt" => unary_number(arguments, f32::sqrt),
        "sin" => unary_number(arguments, |value| value.to_radians().sin()),
        "cos" => unary_number(arguments, |value| value.to_radians().cos()),
        "tan" => unary_number(arguments, |value| value.to_radians().tan()),
        "arcsin" => inverse_trig(arguments, f32::asin),
        "arccos" => inverse_trig(arguments, f32::acos),
        "arctan" => arctan_builtin(arguments),
        "log" => log_builtin(arguments),
        "clamp" => clamp_builtin(arguments, state),
        "lerp" => lerp_builtin(arguments),
        "length_char" => length_char(arguments, state),
        "lowertext" => text_map(arguments, state, str::to_lowercase),
        "uppertext" => text_map(arguments, state, str::to_uppercase),
        "trimtext" => text_map(arguments, state, |value| value.trim().to_owned()),
        "sort_list" => sort_list_builtin(arguments, state),
        "typecacheof" => typecacheof_builtin(arguments, state),
        "ascii2text" => ascii2text(arguments),
        "text2ascii" => text2ascii(arguments, state, false),
        "text2ascii_char" => text2ascii(arguments, state, true),
        "text2num" => text2num(arguments, state),
        "text2path" => text2path(arguments, state),
        "isinf" => numeric_classifier(arguments, f32::is_infinite),
        "isnan" => numeric_classifier(arguments, f32::is_nan),
        "cmptext" => cmptext(arguments, state, false),
        "cmptextEx" => cmptext(arguments, state, true),
        "findtext" => findtext(arguments, state, false, false, false),
        "findtextEx" => findtext(arguments, state, true, false, false),
        "findtext_char" => findtext(arguments, state, false, true, false),
        "findtextEx_char" => findtext(arguments, state, true, true, false),
        "findlasttext" => findtext(arguments, state, false, false, true),
        "findlasttextEx" => findtext(arguments, state, true, false, true),
        "findlasttext_char" => findtext(arguments, state, false, true, true),
        "findlasttextEx_char" => findtext(arguments, state, true, true, true),
        "splittext" => splittext(arguments, state, false),
        "splittext_char" => splittext(arguments, state, true),
        "jointext" => jointext(arguments, state),
        "addtext" => addtext(arguments, state),
        "spantext" => spantext(arguments, state, false, true),
        "spantext_char" => spantext(arguments, state, true, true),
        "nonspantext" => spantext(arguments, state, false, false),
        "nonspantext_char" => spantext(arguments, state, true, false),
        "splicetext" => splicetext(arguments, state, false),
        "splicetext_char" => splicetext(arguments, state, true),
        "get_dist" => get_dist(arguments, state),
        "turn" => turn(arguments, state),
        "astype" => astype(arguments, state),
        "ckey" => ckey(arguments, state),
        "fexists" => fexists(arguments, state),
        "file2text" => file2text(arguments, state),
        "lentext" => lentext(arguments, state),
        "sorttext" => sorttext(arguments, state, false),
        "sorttextEx" | "sortText" => sorttext(arguments, state, true),
        "num2text" => num2text(arguments),
        "list2params" => list2params(arguments, state),
        "params2list" => params2list(arguments, state),
        "json_decode" => json_decode_builtin(arguments, state),
        "json_encode" => json_encode_builtin(arguments, state),
        "md5" => md5_builtin(arguments),
        "image" => image_builtin(arguments, state),
        _ => Err(format!("unknown native DM builtin {name:?}")),
    }
}

fn md5_builtin(arguments: &[Value]) -> Result<Value, String> {
    let Some(Value::Text(text)) = arguments.first() else {
        return Ok(Value::Null);
    };
    Ok(Value::text(format!("{:x}", md5::compute(text.as_bytes()))))
}

fn json_encode_builtin(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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
        Value::Text(text) => Ok(serde_json::Value::String(text.to_string())),
        Value::TypePath(path) => Ok(serde_json::Value::String(path.to_string())),
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

fn json_decode_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let Some(Value::Text(text)) = arguments.first() else {
        return Err("json_decode requires text".to_owned());
    };
    let json: serde_json::Value =
        serde_json::from_str(text).map_err(|error| format!("json_decode failed: {error}"))?;
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

fn lentext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "lentext")?;
    Ok(Value::number(text.len() as f32))
}

fn sorttext(arguments: &[Value], state: &ExecutionState, exact: bool) -> Result<Value, String> {
    if arguments.len() < 2 {
        return Ok(Value::number(0.0));
    }
    let values = arguments
        .iter()
        .map(|value| strict_text(value, state, "sorttext"))
        .collect::<Result<Vec<_>, _>>()?;
    let compare = |left: &str, right: &str| {
        if exact {
            left.cmp(right)
        } else {
            left.to_lowercase().cmp(&right.to_lowercase())
        }
    };
    let ascending = values
        .windows(2)
        .all(|pair| compare(&pair[0], &pair[1]).is_lt());
    let descending = values
        .windows(2)
        .all(|pair| compare(&pair[0], &pair[1]).is_gt());
    Ok(Value::number(if ascending {
        1.0
    } else if descending {
        -1.0
    } else {
        0.0
    }))
}

fn num2text(arguments: &[Value]) -> Result<Value, String> {
    let value = number(&arguments[0], "num2text")?;
    if arguments.len() == 3 {
        let digits = number(&arguments[1], "num2text digits")?.trunc().max(0.0) as usize;
        let radix = number(&arguments[2], "num2text radix")?.trunc() as u32;
        if !(2..=36).contains(&radix) {
            return Err(format!("num2text radix {radix} is outside 2..=36"));
        }
        let negative = value.is_sign_negative();
        let mut integer = value.abs().trunc() as u32;
        let alphabet = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut encoded = Vec::new();
        loop {
            encoded.push(alphabet[(integer % radix) as usize] as char);
            integer /= radix;
            if integer == 0 {
                break;
            }
        }
        while encoded.len() < digits {
            encoded.push('0');
        }
        if negative {
            encoded.push('-');
        }
        encoded.reverse();
        return Ok(Value::text(encoded.into_iter().collect::<String>()));
    }
    let sigfig = arguments.get(1).map_or(Ok(6_usize), |value| {
        number(value, "num2text sigfig").map(|value| value.trunc().max(1.0) as usize)
    })?;
    let plain = value.to_string();
    let significant_digits = plain.chars().filter(char::is_ascii_digit).count();
    if significant_digits <= sigfig || value == 0.0 {
        return Ok(Value::text(plain));
    }
    Ok(Value::text(format!(
        "{:.*e}",
        sigfig.saturating_sub(1),
        value
    )))
}

fn form_encode(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(char::from(byte));
            }
            b' ' => output.push('+'),
            _ => write!(&mut output, "%{byte:02X}").expect("writing to a String cannot fail"),
        }
    }
    output
}

fn form_decode(value: &str) -> Result<String, String> {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3])
                    .map_err(|error| error.to_string())?;
                let byte = u8::from_str_radix(hex, 16)
                    .map_err(|_| format!("invalid parameter escape %{hex}"))?;
                output.push(byte);
                index += 3;
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(output).map_err(|error| format!("parameter text is not UTF-8: {error}"))
}

fn list2params(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Value::List(list_id) = arguments[0] else {
        return Err(format!(
            "list2params requires a list, received {}",
            arguments[0]
        ));
    };
    let list = state
        .heap
        .list(list_id)
        .map_err(|error| error.to_string())?;
    let mut pairs = Vec::with_capacity(list.len());
    for (_, key) in list.positions() {
        let key_text = runtime_text(key, state, "list2params key")?;
        let associated = list.get_key(key).cloned().unwrap_or(Value::Null);
        let value_text = runtime_text(&associated, state, "list2params value")?;
        pairs.push(format!(
            "{}={}",
            form_encode(&key_text),
            form_encode(&value_text)
        ));
    }
    Ok(Value::text(pairs.join("&")))
}

fn params2list(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let params = strict_text(&arguments[0], state, "params2list")?;
    let result = state.heap.allocate_list();
    for part in params.split(['&', ';']) {
        if part.is_empty() {
            continue;
        }
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        let key = Value::text(form_decode(key)?);
        let value = Value::text(form_decode(value)?);
        state
            .heap
            .list_mut(result)
            .map_err(|error| error.to_string())?
            .set_key(key, value);
    }
    Ok(Value::List(result))
}

fn unary_number(arguments: &[Value], operation: impl FnOnce(f32) -> f32) -> Result<Value, String> {
    let value = number(&arguments[0], "numeric builtin")?;
    Ok(Value::number(operation(value)))
}

fn fallback_number(value: &Value) -> f32 {
    match value {
        Value::Number(number) => number.to_f32(),
        Value::Null | Value::Text(_) | Value::TypePath(_) | Value::Datum(_) | Value::List(_) => 0.0,
    }
}

fn inverse_trig(arguments: &[Value], operation: impl FnOnce(f32) -> f32) -> Result<Value, String> {
    let value = fallback_number(&arguments[0]);
    let value = if (-1.0..=1.0).contains(&value) {
        operation(value).to_degrees()
    } else {
        0.0
    };
    Ok(Value::number(value))
}

fn arctan_builtin(arguments: &[Value]) -> Result<Value, String> {
    let first = fallback_number(&arguments[0]);
    let value = if arguments.len() == 1 {
        first.atan().to_degrees()
    } else {
        let second = fallback_number(&arguments[1]);
        second.atan2(first).to_degrees()
    };
    Ok(Value::number(value))
}

fn number(value: &Value, context: &str) -> Result<f32, String> {
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        _ => Err(format!("{context} requires a number, received {value}")),
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Number(number) => number.to_f32() != 0.0,
        Value::Text(text) => !text.is_empty(),
        Value::TypePath(_) | Value::Datum(_) | Value::List(_) => true,
    }
}

fn log_builtin(arguments: &[Value]) -> Result<Value, String> {
    let value = if arguments.len() == 1 {
        number(&arguments[0], "log")?.ln()
    } else {
        let base = number(&arguments[0], "log base")?;
        let value = number(&arguments[1], "log value")?;
        value.log(base)
    };
    Ok(Value::number(value))
}

fn lerp_builtin(arguments: &[Value]) -> Result<Value, String> {
    let start = number(&arguments[0], "lerp start")?;
    let end = number(&arguments[1], "lerp end")?;
    let factor = number(&arguments[2], "lerp factor")?;
    Ok(Value::number(start + (end - start) * factor))
}

/// Implements BYOND's scalar and list `clamp(value, low, high)` forms.
/// Bounds are interchangeable. List input produces a new positional list and
/// skips nonnumeric entries, matching Dream Maker's observable behavior.
fn clamp_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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

fn runtime_text(value: &Value, state: &ExecutionState, context: &str) -> Result<String, String> {
    match value {
        Value::Text(text) => Ok(text.to_string()),
        Value::Null => Ok(String::new()),
        Value::Number(number) => Ok(number.to_f32().to_string()),
        Value::TypePath(path) => Ok(path.to_string()),
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
        Value::List(_) => Err(format!("{context} cannot convert a list to text")),
    }
}

fn qdel_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments.is_empty() {
        return Ok(Value::Null);
    }
    for argument in arguments {
        qdel_value(argument, state).map_err(|error| format!("qdel failed: {error}"))?;
    }
    Ok(Value::Null)
}

fn qdel_value(value: &Value, state: &mut ExecutionState) -> Result<(), String> {
    match value {
        Value::Null => Ok(()),
        Value::Number(_) | Value::Text(_) | Value::TypePath(_) => Ok(()),
        Value::Datum(datum) => state
            .heap_mut()
            .destroy_datum(*datum)
            .map_err(|error| error.to_string())
            .map(|_| ()),
        Value::List(list) => {
            let entries = state
                .heap
                .list(*list)
                .map_err(|error| error.to_string())?
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>();
            for entry in entries {
                qdel_value(&entry, state)?;
            }
            Ok(())
        }
    }
}

fn typecacheof_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let target = arguments
        .first()
        .ok_or_else(|| "typecacheof requires a base type".to_owned())?;
    let target = match target {
        Value::TypePath(path) => path.clone(),
        Value::Text(text) => TypePath::parse(text)
            .map_err(|_| format!("typecacheof requires a type path, received {target}"))?,
        _ => {
            return Err(format!(
                "typecacheof requires a type path, received {target}"
            ));
        }
    };

    let paths = {
        let mut paths = state
            .type_paths()
            .filter(|path| {
                let path = *path;
                path == &target || path.as_str().starts_with(&format!("{}/", target.as_str()))
            })
            .cloned()
            .collect::<Vec<_>>();
        paths.push(target.clone());
        paths.sort_unstable();
        paths
    };

    let result = state.heap_mut().allocate_list();
    let list = state
        .heap_mut()
        .list_mut(result)
        .map_err(|error| error.to_string())?;

    for path in paths {
        let _ = list.set_key(Value::TypePath(path), Value::number(1.0));
    }
    Ok(Value::List(result))
}

fn image_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let image_path = TypePath::parse("/image").expect("\"/image\" is a canonical BYOND type path");
    let image = state.heap_mut().allocate_datum(image_path);
    let datum = state
        .heap_mut()
        .datum_mut(image)
        .map_err(|error| error.to_string())?;

    if let Some(icon) = arguments.first() {
        let _ = datum.set_field(
            FieldName::parse("icon").expect("field name icon"),
            icon.clone(),
        );
    }
    if let Some(location) = arguments.get(1) {
        let _ = datum.set_field(
            FieldName::parse("loc").expect("field name loc"),
            location.clone(),
        );
    }
    if let Some(icon_state) = arguments.get(2) {
        let _ = datum.set_field(
            FieldName::parse("icon_state").expect("field name icon_state"),
            icon_state.clone(),
        );
    }
    if let Some(layer) = arguments.get(3) {
        let _ = datum.set_field(
            FieldName::parse("layer").expect("field name layer"),
            layer.clone(),
        );
    }
    if let Some(direction) = arguments.get(4) {
        let _ = datum.set_field(
            FieldName::parse("dir").expect("field name dir"),
            direction.clone(),
        );
    }

    Ok(Value::Datum(image))
}

fn sort_list_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let list = arguments
        .first()
        .ok_or_else(|| "sort_list requires a list argument".to_owned())?;
    let list = match list {
        Value::List(list) => *list,
        value => return Err(format!("sort_list requires a list, received {value}")),
    };

    let entries = {
        let snapshot = state.heap.list(list).map_err(|error| error.to_string())?;
        if snapshot.associative_len() > 0 {
            return Err("sort_list does not support associative entries yet".to_owned());
        }
        snapshot
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>()
    };

    let mut entries = entries;
    entries.sort_by(|left, right| match (left, right) {
        (Value::Number(left), Value::Number(right)) => left
            .to_f32()
            .partial_cmp(&right.to_f32())
            .unwrap_or(std::cmp::Ordering::Equal),
        _ => {
            let left =
                runtime_text(left, state, "sort_list item").unwrap_or_else(|_| left.to_string());
            let right =
                runtime_text(right, state, "sort_list item").unwrap_or_else(|_| right.to_string());
            left.cmp(&right)
        }
    });

    let list_id = list;
    let list = state
        .heap
        .list_mut(list_id)
        .map_err(|error| error.to_string())?;
    list.resize(0).map_err(|error| error.to_string())?;
    for entry in entries {
        list.add(entry);
    }
    Ok(Value::List(list_id))
}

fn strict_text(value: &Value, state: &ExecutionState, context: &str) -> Result<String, String> {
    match value {
        Value::Text(text) => Ok(text.to_string()),
        _ => Err(format!(
            "{context} requires text, received {}",
            runtime_text(value, state, context)?
        )),
    }
}

fn text_map(
    arguments: &[Value],
    state: &ExecutionState,
    operation: impl FnOnce(&str) -> String,
) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text builtin")?;
    Ok(Value::text(operation(&text)))
}

fn length_char(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let length = match &arguments[0] {
        Value::Null => 0,
        Value::Text(text) => text.chars().count(),
        Value::List(list) => state
            .heap
            .list(*list)
            .map_err(|error| error.to_string())?
            .len(),
        value => {
            return Err(format!(
                "length_char requires text or a list, received {value}"
            ));
        }
    };
    Ok(Value::number(length as f32))
}

fn ascii2text(arguments: &[Value]) -> Result<Value, String> {
    let value = number(&arguments[0], "ascii2text")?;
    if !value.is_finite() {
        return Ok(Value::Null);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let codepoint = value.trunc().max(0.0) as u32;
    Ok(char::from_u32(codepoint)
        .map_or(Value::Null, |character| Value::text(character.to_string())))
}

fn logical_length(text: &str, character_indices: bool) -> usize {
    if character_indices {
        text.chars().count()
    } else {
        text.len()
    }
}

#[allow(clippy::cast_possible_truncation)]
fn signed_position(value: Option<&Value>, default: i64) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => Ok(number.to_f32().trunc() as i64),
        Some(value) => Err(format!("text position requires a number, received {value}")),
    }
}

fn resolve_position(position: i64, length: usize) -> usize {
    let limit = i64::try_from(length)
        .unwrap_or(i64::MAX - 1)
        .saturating_add(1);
    let position = if position < 0 {
        limit.saturating_add(position)
    } else {
        position
    };
    usize::try_from(position.clamp(1, limit)).unwrap_or(usize::MAX)
}

fn byte_offset(text: &str, logical_position_zero_based: usize, character_indices: bool) -> usize {
    if character_indices {
        text.char_indices()
            .nth(logical_position_zero_based)
            .map_or(text.len(), |(offset, _)| offset)
    } else {
        let mut offset = logical_position_zero_based.min(text.len());
        while offset > 0 && !text.is_char_boundary(offset) {
            offset -= 1;
        }
        offset
    }
}

fn text2ascii(
    arguments: &[Value],
    state: &ExecutionState,
    character_indices: bool,
) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text2ascii")?;
    let length = logical_length(&text, character_indices);
    let position = resolve_position(signed_position(arguments.get(1), 1)?, length);
    if position > length {
        return Ok(Value::number(0.0));
    }
    if character_indices {
        let value = text.chars().nth(position - 1).map_or(0, u32::from);
        Ok(Value::number(value as f32))
    } else {
        Ok(Value::number(f32::from(text.as_bytes()[position - 1])))
    }
}

fn text2num(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text2num")?;
    let radix = if let Some(radix) = arguments.get(1) {
        number(radix, "text2num radix")?.trunc() as i32
    } else {
        10
    };
    if !(2..=36).contains(&radix) {
        return Ok(Value::Null);
    }
    let text = text.trim_start();
    if radix == 10 {
        let bytes = text.as_bytes();
        let mut end = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
        let mut saw_digit = false;
        let mut saw_dot = false;
        let mut saw_exp = false;
        while let Some(byte) = bytes.get(end).copied() {
            if byte.is_ascii_digit() {
                saw_digit = true;
                end += 1;
            } else if byte == b'.' && !saw_dot && !saw_exp {
                saw_dot = true;
                end += 1;
            } else if matches!(byte, b'e' | b'E') && saw_digit && !saw_exp {
                saw_exp = true;
                end += 1;
                if matches!(bytes.get(end), Some(b'+' | b'-')) {
                    end += 1;
                }
            } else {
                break;
            }
        }
        if !saw_digit {
            return Ok(Value::Null);
        }
        return text[..end]
            .parse::<f32>()
            .map(Value::number)
            .or(Ok(Value::Null));
    }
    let mut chars = text.char_indices();
    let mut sign = 1_i64;
    let mut start = 0;
    if let Some((_, first)) = chars.next() {
        if first == '-' {
            sign = -1;
            start = 1;
        } else if first == '+' {
            start = 1;
        }
    }
    let mut end = start;
    for (offset, character) in text[start..].char_indices() {
        if character.to_digit(radix as u32).is_none() {
            break;
        }
        end = start + offset + character.len_utf8();
    }
    if end == start {
        return Ok(Value::Null);
    }
    let integer = i64::from_str_radix(&text[start..end], radix as u32).ok();
    Ok(integer.map_or(Value::Null, |value| Value::number((value * sign) as f32)))
}

fn text2path(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text2path")?;
    Ok(state
        .type_paths
        .iter()
        .find(|path| path.as_str() == text)
        .cloned()
        .map_or(Value::Null, Value::TypePath))
}

fn numeric_classifier(
    arguments: &[Value],
    predicate: impl FnOnce(f32) -> bool,
) -> Result<Value, String> {
    Ok(Value::number(f32::from(match &arguments[0] {
        Value::Number(number) => predicate(number.to_f32()),
        _ => false,
    })))
}

fn cmptext(arguments: &[Value], state: &ExecutionState, exact: bool) -> Result<Value, String> {
    let first = strict_text(&arguments[0], state, "cmptext")?;
    for value in &arguments[1..] {
        let value = strict_text(value, state, "cmptext")?;
        let matches = if exact {
            first == value
        } else {
            first.eq_ignore_ascii_case(&value) || first.to_lowercase() == value.to_lowercase()
        };
        if !matches {
            return Ok(Value::number(0.0));
        }
    }
    Ok(Value::number(1.0))
}

fn text_region(text: &str, start: i64, end: i64, character_indices: bool) -> (usize, usize, usize) {
    let length = logical_length(text, character_indices);
    let start = resolve_position(start, length);
    let end = if end == 0 {
        length.saturating_add(1)
    } else {
        resolve_position(end, length)
    };
    let (start, end) = if end < start {
        (end, start)
    } else {
        (start, end)
    };
    let start_byte = byte_offset(text, start.saturating_sub(1), character_indices);
    let end_byte = byte_offset(text, end.saturating_sub(1), character_indices);
    (start_byte, end_byte, start)
}

fn find_match(text: &str, needle: &str, exact: bool, reverse: bool) -> Option<usize> {
    let matches_at = |offset: usize| {
        let tail = &text[offset..];
        if exact {
            tail.starts_with(needle)
        } else {
            tail.get(..needle.len())
                .is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
                || tail.to_lowercase().starts_with(&needle.to_lowercase())
        }
    };
    let mut offsets = text
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()));
    if reverse {
        offsets
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .find(|offset| matches_at(*offset))
    } else {
        offsets.find(|offset| matches_at(*offset))
    }
}

fn findtext(
    arguments: &[Value],
    state: &ExecutionState,
    exact: bool,
    character_indices: bool,
    reverse: bool,
) -> Result<Value, String> {
    let haystack = strict_text(&arguments[0], state, "findtext haystack")?;
    if matches!(arguments[1], Value::Datum(_)) {
        return Err(
            "regex needles in findtext are not yet supported by the headless VM".to_owned(),
        );
    }
    let needle = strict_text(&arguments[1], state, "findtext needle")?;
    if reverse {
        let length = logical_length(&haystack, character_indices);
        let start = signed_position(arguments.get(2), 0)?;
        let start = if start == 0 {
            length.saturating_add(1)
        } else {
            resolve_position(start, length)
        };
        let end = resolve_position(signed_position(arguments.get(3), 1)?, length);
        if start < end {
            return Ok(Value::number(0.0));
        }
        let region_start = byte_offset(&haystack, end.saturating_sub(1), character_indices);
        let region_end = byte_offset(&haystack, start.saturating_sub(1), character_indices);
        let region = &haystack[region_start..region_end];
        let Some(found) = find_match(region, &needle, exact, true) else {
            return Ok(Value::number(0.0));
        };
        let byte = region_start + found;
        let position = if character_indices {
            haystack[..byte].chars().count() + 1
        } else {
            byte + 1
        };
        return Ok(Value::number(position as f32));
    }
    let start = signed_position(arguments.get(2), 1)?;
    let end = signed_position(arguments.get(3), 0)?;
    let (region_start, region_end, _) = text_region(&haystack, start, end, character_indices);
    let region = &haystack[region_start..region_end];
    let Some(found) = find_match(region, &needle, exact, false) else {
        return Ok(Value::number(0.0));
    };
    let byte = region_start + found;
    let position = if character_indices {
        haystack[..byte].chars().count() + 1
    } else {
        byte + 1
    };
    Ok(Value::number(position as f32))
}

fn splittext(
    arguments: &[Value],
    state: &mut ExecutionState,
    character_indices: bool,
) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "splittext text")?;
    if matches!(arguments[1], Value::Datum(_)) {
        return Err(
            "regex delimiters in splittext are not yet supported by the headless VM".to_owned(),
        );
    }
    let delimiter = strict_text(&arguments[1], state, "splittext delimiter")?;
    let start = signed_position(arguments.get(2), 1)?;
    let end = signed_position(arguments.get(3), 0)?;
    let include_delimiters = arguments.get(4).is_some_and(truthy);
    let (start, end, _) = text_region(&text, start, end, character_indices);
    let target = &text[start..end];
    let list = state.heap.allocate_list();
    let mut output = Vec::new();
    if delimiter.is_empty() {
        output.extend(target.chars().map(|character| character.to_string()));
    } else {
        let mut cursor = 0;
        while let Some(found) = target[cursor..].find(&delimiter) {
            let found = cursor + found;
            output.push(target[cursor..found].to_owned());
            if include_delimiters {
                output.push(delimiter.clone());
            }
            cursor = found + delimiter.len();
        }
        output.push(target[cursor..].to_owned());
    }
    let entries = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for item in output {
        entries.add(Value::text(item));
    }
    Ok(Value::List(list))
}

fn jointext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Value::List(list) = arguments[0] else {
        return Err(format!(
            "jointext requires a list, received {}",
            arguments[0]
        ));
    };
    let glue = runtime_text(&arguments[1], state, "jointext glue")?;
    let list = state.heap.list(list).map_err(|error| error.to_string())?;
    let length = list.len();
    let start = resolve_position(signed_position(arguments.get(2), 1)?, length);
    let end_arg = signed_position(arguments.get(3), 0)?;
    let end = if end_arg == 0 {
        length.saturating_add(1)
    } else {
        resolve_position(end_arg, length)
    };
    let mut items = Vec::new();
    for index in start..end.min(length.saturating_add(1)) {
        let value = list.get(index).map_err(|error| error.to_string())?;
        items.push(runtime_text(value, state, "jointext item")?);
    }
    Ok(Value::text(items.join(&glue)))
}

fn addtext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let mut output = String::new();
    for value in arguments {
        output.push_str(&strict_text(value, state, "addtext")?);
    }
    Ok(Value::text(output))
}

fn spantext(
    arguments: &[Value],
    state: &ExecutionState,
    character_indices: bool,
    matching: bool,
) -> Result<Value, String> {
    let haystack = strict_text(&arguments[0], state, "spantext haystack")?;
    let needles = strict_text(&arguments[1], state, "spantext needles")?;
    let length = logical_length(&haystack, character_indices);
    let start = resolve_position(signed_position(arguments.get(2), 1)?, length);
    let start_byte = byte_offset(&haystack, start.saturating_sub(1), character_indices);
    let mut count = 0usize;
    for character in haystack[start_byte..].chars() {
        let contains = needles.contains(character);
        if contains != matching {
            break;
        }
        count += if character_indices {
            1
        } else {
            character.len_utf8()
        };
    }
    Ok(Value::number(count as f32))
}

fn splicetext(
    arguments: &[Value],
    state: &ExecutionState,
    character_indices: bool,
) -> Result<Value, String> {
    let source = strict_text(&arguments[0], state, "splicetext text")?;
    let start = signed_position(arguments.get(1), 1)?;
    let end = signed_position(arguments.get(2), 0)?;
    let replacement = strict_text(&arguments[3], state, "splicetext replacement")?;
    let (start, end, _) = text_region(&source, start, end, character_indices);
    Ok(Value::text(format!(
        "{}{}{}",
        &source[..start],
        replacement,
        &source[end..]
    )))
}

fn datum_coordinates(state: &ExecutionState, value: &Value) -> Option<(f32, f32, f32)> {
    let Value::Datum(datum) = value else {
        return None;
    };
    let datum = state.heap.datum(*datum).ok()?;
    let coordinate = |name: &str| {
        datum
            .field(&FieldName::parse(name).expect("coordinate name is valid"))
            .ok()?
            .as_number()
    };
    Some((coordinate("x")?, coordinate("y")?, coordinate("z")?))
}

fn get_dist(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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

pub(super) fn is_subtype(state: &ExecutionState, candidate: &TypePath, target: &TypePath) -> bool {
    if candidate == target {
        return true;
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

fn fallback_parent(path: &TypePath) -> Option<TypePath> {
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

fn astype(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Value::TypePath(target) = &arguments[1] else {
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

fn turn(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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

fn ckey(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let key = strict_text(&arguments[0], state, "ckey")?;
    Ok(Value::text(
        key.chars()
            .filter(|character| character.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>(),
    ))
}

#[derive(Clone)]
struct ListOperatorEntry {
    key: Value,
    associated: Option<Value>,
}

fn list_operator_snapshot(
    list: ListId,
    state: &ExecutionState,
) -> Result<Vec<ListOperatorEntry>, String> {
    let list = state.heap.list(list).map_err(|error| error.to_string())?;
    Ok(list
        .positions()
        .map(|(_, key)| {
            let associated = list.get_key(key).ok().cloned();
            ListOperatorEntry {
                key: key.clone(),
                associated,
            }
        })
        .collect())
}

fn add_operator_entry(
    list: ListId,
    entry: ListOperatorEntry,
    state: &mut ExecutionState,
    only_if_absent: bool,
) -> Result<(), String> {
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    if only_if_absent && target.contains(&entry.key) {
        return Ok(());
    }
    if let Some(associated) = entry.associated {
        target.set_key(entry.key, associated);
    } else {
        target.add(entry.key);
    }
    Ok(())
}

fn remove_all_operator_matches(
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

fn operator_rhs_entries(
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

pub(super) fn execute_list_binary_operator(
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
            for entry in operator_rhs_entries(right, state)? {
                state
                    .heap
                    .list_mut(result)
                    .map_err(|error| error.to_string())?
                    .remove_last(&entry.key);
            }
            Ok(Value::List(result))
        }
        "|" => {
            let result = state.heap.allocate_list();
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

pub(super) fn execute_list_compound_operator(
    operator: CompoundAssignmentOperator,
    left: ListId,
    right: &Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match operator {
        CompoundAssignmentOperator::Add => {
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(left, entry, state, false)?;
            }
        }
        CompoundAssignmentOperator::Subtract => {
            for entry in operator_rhs_entries(right, state)? {
                state
                    .heap
                    .list_mut(left)
                    .map_err(|error| error.to_string())?
                    .remove_last(&entry.key);
            }
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
    Ok(Value::List(left))
}

pub(super) fn execute_list_method(
    name: &str,
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Option<Result<Value, String>> {
    Some(match name {
        "Add" => list_add(list, arguments, state),
        "Copy" => list_copy(list, arguments, state),
        "Cut" => list_cut(list, arguments, state),
        "Find" => list_find(list, arguments, state),
        "Insert" => list_insert(list, arguments, state),
        "Join" => list_join(list, arguments, state),
        "Remove" => list_remove(list, arguments, state, false),
        "RemoveAll" => list_remove(list, arguments, state, true),
        "Splice" => list_splice(list, arguments, state),
        "Swap" => list_swap(list, arguments, state),
        _ => return None,
    })
}

fn list_integer(value: Option<&Value>, default: i64, context: &str) -> Result<i64, String> {
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

fn list_boundary(value: i64, len: usize, zero_is_end: bool) -> Result<usize, String> {
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

fn splice_boundary(value: i64, len: usize, zero_is_end: bool) -> usize {
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

fn flattened_list_arguments(
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

fn list_add(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.is_empty() {
        return Err("list.Add requires at least one item".to_owned());
    }
    let values = flattened_list_arguments(arguments, state)?;
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for value in values {
        target.add(value);
    }
    Ok(Value::Null)
}

fn list_copy(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() > 2 {
        return Err("list.Copy accepts Start and End only".to_owned());
    }
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
    Ok(Value::List(result))
}

fn list_cut(
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
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .cut_range(start, end)
        .map_err(|error| error.to_string())?;
    Ok(Value::Null)
}

fn list_find(list: ListId, arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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

fn list_insert(
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
    Ok(Value::number(index as f32))
}

fn list_join(list: ListId, arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    if arguments.is_empty() || arguments.len() > 3 {
        return Err("list.Join requires Glue and optional Start/End".to_owned());
    }
    let glue = runtime_text(&arguments[0], state, "list.Join Glue")?;
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

fn list_remove_once(
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

fn list_remove(
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
    if all {
        let mut total = 0usize;
        loop {
            let removed = list_remove_once(list, arguments, state)?;
            total += removed;
            if removed == 0 {
                break;
            }
        }
        Ok(Value::number(total as f32))
    } else {
        Ok(Value::number(f32::from(
            list_remove_once(list, arguments, state)? > 0,
        )))
    }
}

fn list_splice(
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
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .cut_range(start, end)
        .map_err(|error| error.to_string())?;
    if arguments.len() <= 2 {
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
    Ok(Value::Null)
}

fn list_swap(
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

fn resolved_file_path(
    arguments: &[Value],
    state: &ExecutionState,
    context: &str,
) -> Result<PathBuf, String> {
    let path = strict_text(&arguments[0], state, context)?;
    let path = PathBuf::from(path);
    if path.is_absolute() {
        Ok(path)
    } else if let Some(root) = &state.project_root {
        Ok(root.join(path))
    } else {
        Ok(path)
    }
}

fn fexists(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let path = resolved_file_path(arguments, state, "fexists")?;
    Ok(Value::number(f32::from(path.exists())))
}

fn file2text(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let path = resolved_file_path(arguments, state, "file2text")?;
    match fs::read_to_string(path) {
        Ok(text) => Ok(Value::text(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(error) => Err(format!("file2text failed: {error}")),
    }
}

#[cfg(test)]
mod json_md5_tests {
    use super::*;

    fn encoded(value: Value, state: &ExecutionState) -> String {
        let Value::Text(text) = json_encode_builtin(&[value], state).expect("JSON should encode")
        else {
            panic!("json_encode must return text");
        };
        text.to_string()
    }

    #[test]
    fn json_encodes_dm_scalars_and_special_numbers() {
        let state = ExecutionState::new();
        assert_eq!(encoded(Value::Null, &state), "null");
        assert_eq!(encoded(Value::number(7.0), &state), "7");
        assert_eq!(encoded(Value::number(15.5), &state), "15.5");
        assert_eq!(encoded(Value::text("A\nB"), &state), r#""A\nB""#);
        assert_eq!(
            encoded(Value::number(f32::NAN), &state),
            r#"{"__number__":"NaN"}"#
        );
        assert_eq!(
            encoded(Value::number(f32::INFINITY), &state),
            r#"{"__number__":"Infinity"}"#
        );
    }

    #[test]
    fn json_encodes_positional_associative_and_pretty_lists() {
        let mut state = ExecutionState::new();
        let positional = state.heap.allocate_list();
        state
            .heap
            .list_mut(positional)
            .unwrap()
            .add(Value::number(1.0));
        state
            .heap
            .list_mut(positional)
            .unwrap()
            .add(Value::text("two"));
        assert_eq!(encoded(Value::List(positional), &state), r#"[1,"two"]"#);

        let associative = state.heap.allocate_list();
        state
            .heap
            .list_mut(associative)
            .unwrap()
            .set_key(Value::text("name"), Value::text("fridge"));
        state
            .heap
            .list_mut(associative)
            .unwrap()
            .set_key(Value::text("power"), Value::number(12.0));
        assert_eq!(
            encoded(Value::List(associative), &state),
            r#"{"name":"fridge","power":12}"#
        );
        let Value::Text(pretty) =
            json_encode_builtin(&[Value::List(associative), Value::number(1.0)], &state).unwrap()
        else {
            panic!("pretty JSON must be text");
        };
        assert!(pretty.contains('\n'));
    }

    #[test]
    fn json_decodes_arrays_objects_booleans_and_special_numbers() {
        let mut state = ExecutionState::new();
        let decoded =
            json_decode_builtin(&[Value::text(r#"{"a":[true,null,2.5]}"#)], &mut state).unwrap();
        assert_eq!(encoded(decoded, &state), r#"{"a":[1,null,2.5]}"#);
        let special =
            json_decode_builtin(&[Value::text(r#"{"__number__":"-Infinity"}"#)], &mut state)
                .unwrap();
        assert!(special.as_number().unwrap().is_infinite());
        assert!(special.as_number().unwrap().is_sign_negative());
    }

    #[test]
    fn md5_hashes_text_bytes_and_rejects_non_text_values() {
        assert_eq!(
            md5_builtin(&[Value::text("md5_test")]).unwrap(),
            Value::text("c74318b61a3024520c466f828c043c79")
        );
        assert_eq!(md5_builtin(&[Value::number(5.0)]).unwrap(), Value::Null);
        assert_eq!(md5_builtin(&[]).unwrap(), Value::Null);
        assert_eq!(encoded(Value::Null, &ExecutionState::new()), "null");
    }
}
