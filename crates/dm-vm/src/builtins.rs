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

use std::fs;
use std::path::PathBuf;

use dm_value::{FieldName, TypePath, Value};

use super::ExecutionState;

pub(super) fn standard_builtin_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "abs" | "ceil" | "floor" | "fract" | "trunc" | "sign" | "sqrt" | "sin" | "cos" | "tan"
        | "length_char" | "lowertext" | "uppertext" | "trimtext" | "ascii2text" | "text2path"
        | "isinf" | "isnan" | "ckey" | "fexists" | "file2text" => (1, 1),
        "log" | "text2ascii" | "text2ascii_char" | "text2num" => (1, 2),
        "lerp" => (3, 3),
        "cmptext" | "cmptextEx" => (1, usize::MAX),
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
        "addtext" => (0, usize::MAX),
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
        "log" => log_builtin(arguments),
        "lerp" => lerp_builtin(arguments),
        "length_char" => length_char(arguments, state),
        "lowertext" => text_map(arguments, state, str::to_lowercase),
        "uppertext" => text_map(arguments, state, str::to_uppercase),
        "trimtext" => text_map(arguments, state, |value| value.trim().to_owned()),
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
        _ => Err(format!("unknown native DM builtin {name:?}")),
    }
}

fn unary_number(arguments: &[Value], operation: impl FnOnce(f32) -> f32) -> Result<Value, String> {
    let value = number(&arguments[0], "numeric builtin")?;
    Ok(Value::number(operation(value)))
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
