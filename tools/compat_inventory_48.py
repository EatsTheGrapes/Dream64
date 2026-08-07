from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}\n---OLD---\n{old[:800]}")
    return text.replace(old, new, 1)


builtins = r'''//! Native implementations of documented BYOND global procedures.
//!
//! These routines are deliberately runtime primitives rather than injected DM
//! source when their behavior depends on host state, type metadata, or precise
//! text-indexing semantics.

use std::fs;
use std::path::PathBuf;

use dm_value::{FieldName, TypePath, Value};

use super::ExecutionState;

pub(super) fn standard_builtin_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "abs" | "ceil" | "floor" | "fract" | "trunc" | "sign" | "sqrt" | "sin"
        | "cos" | "tan" | "length_char" | "lowertext" | "uppertext" | "trimtext"
        | "ascii2text" | "text2path" | "isinf" | "isnan" | "ckey" | "fexists"
        | "file2text" => (1, 1),
        "log" | "text2ascii" | "text2ascii_char" | "text2num" => (1, 2),
        "lerp" => (3, 3),
        "cmptext" | "cmptextEx" => (1, usize::MAX),
        "findtext" | "findtextEx" | "findtext_char" | "findtextEx_char"
        | "findlasttext" | "findlasttextEx" | "findlasttext_char"
        | "findlasttextEx_char" => (2, 4),
        "splittext" | "splittext_char" => (2, 5),
        "jointext" => (2, 4),
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
            let datum = state.heap.datum(*datum).map_err(|error| error.to_string())?;
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
        _ => Err(format!("{context} requires text, received {}", runtime_text(value, state, context)?)),
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
        Value::List(list) => state.heap.list(*list).map_err(|error| error.to_string())?.len(),
        value => return Err(format!("length_char requires text or a list, received {value}")),
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
    Ok(char::from_u32(codepoint).map_or(Value::Null, |character| Value::text(character.to_string())))
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
    let limit = i64::try_from(length).unwrap_or(i64::MAX - 1).saturating_add(1);
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

fn numeric_classifier(arguments: &[Value], predicate: impl FnOnce(f32) -> bool) -> Result<Value, String> {
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

fn text_region(
    text: &str,
    start: i64,
    end: i64,
    character_indices: bool,
) -> (usize, usize, usize) {
    let length = logical_length(text, character_indices);
    let start = resolve_position(start, length);
    let end = if end == 0 {
        length.saturating_add(1)
    } else {
        resolve_position(end, length)
    };
    let (start, end) = if end < start { (end, start) } else { (start, end) };
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
            tail.get(..needle.len()).is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
                || tail.to_lowercase().starts_with(&needle.to_lowercase())
        }
    };
    let offsets = text
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()));
    if reverse {
        offsets.collect::<Vec<_>>().into_iter().rev().find(|offset| matches_at(*offset))
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
        return Err("regex needles in findtext are not yet supported by the headless VM".to_owned());
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
        return Err("regex delimiters in splittext are not yet supported by the headless VM".to_owned());
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
    let entries = state.heap.list_mut(list).map_err(|error| error.to_string())?;
    for item in output {
        entries.add(Value::text(item));
    }
    Ok(Value::List(list))
}

fn jointext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Value::List(list) = arguments[0] else {
        return Err(format!("jointext requires a list, received {}", arguments[0]));
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
        count += if character_indices { 1 } else { character.len_utf8() };
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
    Ok(Value::text(format!("{}{}{}", &source[..start], replacement, &source[end..])))
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
    if matches!((&arguments[0], &arguments[1]), (Value::Datum(left), Value::Datum(right)) if left == right) {
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
        "/area" | "/turf" => Some("/atom"),
        "/atom/movable" => Some("/atom"),
        "/atom" => Some("/datum"),
        "/datum" | "/world" | "/list" | "/client" => None,
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
    let direction = number(&arguments[0], "turn direction")?.trunc() as i32;
    let angle = number(&arguments[1], "turn angle")?;
    let steps = (angle / 45.0).trunc() as i32;
    if steps == 0 {
        return Ok(Value::number(direction as f32));
    }
    const DIRECTIONS: [i32; 8] = [1, 9, 8, 10, 2, 6, 4, 5];
    let index = DIRECTIONS.iter().position(|candidate| *candidate == direction);
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

fn resolved_file_path(arguments: &[Value], state: &ExecutionState, context: &str) -> Result<PathBuf, String> {
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
'''

Path("crates/dm-vm/src/builtins.rs").write_text(builtins)

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
text = replace_once(
    text,
    "#![cfg_attr(not(test), deny(missing_docs))]\n\nuse std::collections::{BTreeMap, HashMap};\nuse std::fmt;\nuse std::sync::Arc;\n",
    "#![cfg_attr(not(test), deny(missing_docs))]\n\nmod builtins;\n\nuse std::collections::{BTreeMap, HashMap};\nuse std::fmt;\nuse std::path::PathBuf;\nuse std::sync::Arc;\n\nuse builtins::{execute_standard_builtin, is_subtype, standard_builtin_arity};\n",
    "dm-vm module imports",
)

text = replace_once(
    text,
    "    /// Enumerates every materialized turf in an inclusive 3D rectangular block.\n    Block {\n",
    "    /// Executes a documented BYOND global procedure handled by the native runtime.\n    StandardBuiltin {\n        /// Canonical global procedure name.\n        name: String,\n        /// Number of already-evaluated arguments.\n        argument_count: u16,\n    },\n    /// Reads a field's compile-time initial value from a datum or type path.\n    InitialField(FieldName),\n    /// Enumerates every materialized turf in an inclusive 3D rectangular block.\n    Block {\n",
    "native builtin instructions",
)
text = replace_once(
    text,
    "    /// Numeric multiplication.\n    Multiply,\n    /// Numeric division.\n",
    "    /// Numeric multiplication.\n    Multiply,\n    /// Numeric exponentiation (`**`).\n    Power,\n    /// Numeric division.\n",
    "power instruction",
)

text = replace_once(
    text,
    "    CopyText {\n        arguments: Vec<Self>,\n        character_indices: bool,\n    },\n    Block {\n",
    "    CopyText {\n        arguments: Vec<Self>,\n        character_indices: bool,\n    },\n    StandardBuiltin {\n        name: String,\n        arguments: Vec<Self>,\n    },\n    Initial(Box<Self>),\n    Block {\n",
    "expression builtin variants",
)

text = replace_once(
    text,
    "fn dm_builtin_numeric_constant(identifier: &str) -> Option<f32> {\n",
    "fn dm_builtin_text_constant(identifier: &str) -> Option<&'static str> {\n    match identifier {\n        \"UNIX\" => Some(\"UNIX\"),\n        \"MS_WINDOWS\" => Some(\"MS Windows\"),\n        \"MALE\" => Some(\"male\"),\n        \"FEMALE\" => Some(\"female\"),\n        \"NEUTER\" => Some(\"neuter\"),\n        \"PLURAL\" => Some(\"plural\"),\n        _ => None,\n    }\n}\n\nfn dm_builtin_numeric_constant(identifier: &str) -> Option<f32> {\n",
    "text builtin constants",
)
text = replace_once(
    text,
    "            TokenKind::Identifier(identifier)\n                if let Some(value) = dm_builtin_numeric_constant(identifier) =>\n            {\n                Ok(Expression::Number(DmNumberBits::from_f32(value)))\n            }\n            TokenKind::Identifier(identifier) if identifier == \"src\" => Ok(Expression::Src),\n",
    "            TokenKind::Identifier(identifier)\n                if let Some(value) = dm_builtin_numeric_constant(identifier) =>\n            {\n                Ok(Expression::Number(DmNumberBits::from_f32(value)))\n            }\n            TokenKind::Identifier(identifier)\n                if let Some(value) = dm_builtin_text_constant(identifier) =>\n            {\n                Ok(Expression::Text(value.to_owned()))\n            }\n            TokenKind::Identifier(identifier) if identifier == \"src\" => Ok(Expression::Src),\n",
    "parse text constants",
)

text = replace_once(
    text,
    "                } else if identifier == \"regex\" {\n",
    "                } else if identifier == \"initial\" {\n                    let mut arguments = self.parse_call_arguments()?;\n                    if arguments.len() != 1 {\n                        return Err(compile_error(format!(\n                            \"initial requires exactly one variable reference, received {} arguments\",\n                            arguments.len()\n                        )));\n                    }\n                    Ok(Expression::Initial(Box::new(arguments.pop().expect(\"validated initial argument\"))))\n                } else if identifier == \"regex\" {\n",
    "parse initial builtin",
)
text = replace_once(
    text,
    "                } else if identifier == \"nameof\" {\n                    self.parse_nameof_expression()\n                } else {\n                    let arguments = self.parse_call_arguments()?;\n                    Ok(Expression::Call {\n                        procedure: identifier.clone(),\n                        arguments,\n                    })\n                }\n",
    "                } else if identifier == \"nameof\" {\n                    self.parse_nameof_expression()\n                } else if let Some((minimum, maximum)) = standard_builtin_arity(identifier) {\n                    let arguments = self.parse_call_arguments()?;\n                    if arguments.len() < minimum || arguments.len() > maximum {\n                        return Err(compile_error(format!(\n                            \"{identifier} received {} arguments; expected {minimum} through {maximum}\",\n                            arguments.len()\n                        )));\n                    }\n                    Ok(Expression::StandardBuiltin {\n                        name: identifier.clone(),\n                        arguments,\n                    })\n                } else {\n                    let arguments = self.parse_call_arguments()?;\n                    Ok(Expression::Call {\n                        procedure: identifier.clone(),\n                        arguments,\n                    })\n                }\n",
    "parse standard builtin catalog",
)

text = text.replace(
    "                        TypePredicateKind::IsType => (1..=2).contains(&arguments.len()),\n                        // BYOND `isloc` is variadic and succeeds only when\n                        // every supplied value is a location.\n                        TypePredicateKind::IsLoc => !arguments.is_empty(),\n                        _ => arguments.len() == 1,\n",
    "                        TypePredicateKind::IsType | TypePredicateKind::IsPath => {\n                            (1..=2).contains(&arguments.len())\n                        }\n                        // BYOND's location classifiers accept multiple values\n                        // and succeed only when every supplied value matches.\n                        TypePredicateKind::IsLoc\n                        | TypePredicateKind::IsMovable\n                        | TypePredicateKind::IsTurf => !arguments.is_empty(),\n                        _ => arguments.len() == 1,\n",
    1,
)

text = replace_once(
    text,
    "            let right = self.parse_binary(precedence + 1)?;\n",
    "            let right_precedence = if operator == \"**\" { precedence } else { precedence + 1 };\n            let right = self.parse_binary(right_precedence)?;\n",
    "right associative power",
)
text = replace_once(
    text,
    "        b\"+\" | b\"-\" => Some(8),\n        b\"*\" | b\"/\" | b\"%\" => Some(9),\n",
    "        b\"+\" | b\"-\" => Some(8),\n        b\"*\" | b\"/\" | b\"%\" => Some(9),\n        b\"**\" => Some(10),\n",
    "power precedence",
)

text = replace_once(
    text,
    "        Expression::Call {\n            procedure,\n            arguments,\n        } => {\n",
    "        Expression::StandardBuiltin { name, arguments } => {\n            let argument_count = u16::try_from(arguments.len())\n                .map_err(|_| compile_error(\"native builtin has more than 65535 arguments\"))?;\n            for argument in arguments {\n                emit_expression(argument, locals, instructions, procedures)?;\n            }\n            instructions.push(Instruction::StandardBuiltin {\n                name: name.clone(),\n                argument_count,\n            });\n        }\n        Expression::Initial(reference) => match reference.as_ref() {\n            Expression::Field { receiver, name } => {\n                emit_expression(receiver, locals, instructions, procedures)?;\n                instructions.push(Instruction::InitialField(name.clone()));\n            }\n            Expression::Local(name) => {\n                let field = locals\n                    .src_field(name)\n                    .ok_or_else(|| compile_error(format!(\"initial target {name:?} is not an instance field\")))?;\n                instructions.push(Instruction::LoadSrc);\n                instructions.push(Instruction::InitialField(field.clone()));\n            }\n            _ => return Err(compile_error(\"initial requires a field reference\")),\n        },\n        Expression::Call {\n            procedure,\n            arguments,\n        } => {\n",
    "emit native and initial expressions",
)
text = replace_once(
    text,
    "                \"*\" => Instruction::Multiply,\n                \"/\" => Instruction::Divide,\n",
    "                \"*\" => Instruction::Multiply,\n                \"**\" => Instruction::Power,\n                \"/\" => Instruction::Divide,\n",
    "emit power",
)

text = replace_once(
    text,
    "        Expression::Call { arguments, .. }\n        | Expression::Regex { arguments }\n",
    "        Expression::Call { arguments, .. }\n        | Expression::StandardBuiltin { arguments, .. }\n        | Expression::Regex { arguments }\n",
    "initializer native arguments",
)
text = replace_once(
    text,
    "        Expression::Length { value }\n        | Expression::Ref { value }\n        | Expression::TypesOf { value } => {\n",
    "        Expression::Length { value }\n        | Expression::Ref { value }\n        | Expression::TypesOf { value }\n        | Expression::Initial(value) => {\n",
    "initializer initial recursion",
)

text = replace_once(
    text,
    "#[derive(Default)]\npub struct ExecutionState {\n    heap: ValueHeap,\n    globals: BTreeMap<FieldName, Value>,\n    type_paths: Arc<std::collections::BTreeSet<TypePath>>,\n    random_state: u64,\n}\n",
    "#[derive(Default)]\npub struct ExecutionState {\n    heap: ValueHeap,\n    globals: BTreeMap<FieldName, Value>,\n    type_paths: Arc<std::collections::BTreeSet<TypePath>>,\n    type_parents: Arc<BTreeMap<TypePath, Option<TypePath>>>,\n    initial_values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,\n    project_root: Option<Arc<PathBuf>>,\n    random_state: u64,\n}\n",
    "execution state catalogs",
)
text = replace_once(
    text,
    "            type_paths: Arc::new(std::collections::BTreeSet::new()),\n            random_state: 0,\n",
    "            type_paths: Arc::new(std::collections::BTreeSet::new()),\n            type_parents: Arc::new(BTreeMap::new()),\n            initial_values: Arc::new(BTreeMap::new()),\n            project_root: None,\n            random_state: 0,\n",
    "execution state from heap",
)
text = replace_once(
    text,
    "    /// Iterates the canonical type catalog in lexical path order.\n    pub fn type_paths(&self) -> impl Iterator<Item = &TypePath> {\n        self.type_paths.iter()\n    }\n\n    /// Iterates globals in canonical field-name order for snapshots.\n",
    "    /// Iterates the canonical type catalog in lexical path order.\n    pub fn type_paths(&self) -> impl Iterator<Item = &TypePath> {\n        self.type_paths.iter()\n    }\n\n    /// Replaces the runtime type-parent catalog used by subtype and parent_type lookups.\n    pub fn set_type_parents(&mut self, parents: BTreeMap<TypePath, Option<TypePath>>) {\n        self.type_parents = Arc::new(parents);\n    }\n\n    /// Replaces effective compile-time initial field values for every runtime type.\n    pub fn set_initial_values(\n        &mut self,\n        values: BTreeMap<TypePath, BTreeMap<FieldName, Value>>,\n    ) {\n        self.initial_values = Arc::new(values);\n    }\n\n    /// Sets the project root used by BYOND filesystem procedures such as fexists().\n    pub fn set_project_root(&mut self, root: PathBuf) {\n        self.project_root = Some(Arc::new(root));\n    }\n\n    /// Returns a type's runtime parent when the catalog contains that type.\n    #[must_use]\n    pub fn type_parent(&self, path: &TypePath) -> Option<&TypePath> {\n        self.type_parents.get(path).and_then(Option::as_ref)\n    }\n\n    /// Returns one effective compile-time initial value when available.\n    #[must_use]\n    pub fn initial_value(&self, path: &TypePath, field: &FieldName) -> Option<&Value> {\n        self.initial_values.get(path).and_then(|fields| fields.get(field))\n    }\n\n    /// Returns the project root used for relative filesystem paths.\n    #[must_use]\n    pub fn project_root(&self) -> Option<&std::path::Path> {\n        self.project_root.as_deref().map(PathBuf::as_path)\n    }\n\n    /// Iterates globals in canonical field-name order for snapshots.\n",
    "execution state catalog methods",
)

text = replace_once(
    text,
    "            Instruction::Length => {\n",
    "            Instruction::StandardBuiltin {\n                name,\n                argument_count,\n            } => {\n                let count = usize::from(argument_count);\n                if count > frames[frame_index].stack.len() {\n                    return Err(execution_error(module, &frames, \"bytecode stack underflow\"));\n                }\n                let arguments = {\n                    let stack = &mut frames[frame_index].stack;\n                    stack.split_off(stack.len() - count)\n                };\n                let value = execute_standard_builtin(&name, &arguments, state)\n                    .map_err(|message| execution_error(module, &frames, message))?;\n                frames[frame_index].stack.push(value);\n            }\n            Instruction::Length => {\n",
    "execute native builtins",
)

old_load = '''            Instruction::LoadField(name) => {
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let datum = match datum_receiver(&receiver, "field read") {
                    Ok(datum) => datum,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                // `datum.type` is a built-in, read-only field.  It is not a
                // materialized user field: its value always reflects the
                // heap datum's canonical runtime type.
                let value = if name.as_str() == "type" {
                    match state.heap.datum(datum) {
                        Ok(datum) => Value::TypePath(datum.type_path().clone()),
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                } else {
                    match state.heap.datum_field(datum, &name) {
                        Ok(value) => value.clone(),
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                };
                frames[frame_index].stack.push(value);
            }
'''
new_load = '''            Instruction::LoadField(name) => {
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let value = match receiver {
                    Value::TypePath(path) if name.as_str() == "parent_type" => state
                        .type_parent(&path)
                        .cloned()
                        .map_or(Value::Null, Value::TypePath),
                    Value::Datum(datum) => {
                        let runtime_type = match state.heap.datum(datum) {
                            Ok(datum) => datum.type_path().clone(),
                            Err(error) => {
                                return Err(execution_error(module, &frames, error.to_string()));
                            }
                        };
                        if name.as_str() == "type" {
                            Value::TypePath(runtime_type)
                        } else if name.as_str() == "parent_type" {
                            state
                                .type_parent(&runtime_type)
                                .cloned()
                                .map_or(Value::Null, Value::TypePath)
                        } else {
                            match state.heap.datum_field(datum, &name) {
                                Ok(value) => value.clone(),
                                Err(error) => {
                                    return Err(execution_error(module, &frames, error.to_string()));
                                }
                            }
                        }
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("field read requires a datum, received {value}"),
                        ));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::InitialField(name) => {
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let runtime_type = match receiver {
                    Value::TypePath(path) => path,
                    Value::Datum(datum) => match state.heap.datum(datum) {
                        Ok(datum) => datum.type_path().clone(),
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    },
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("initial requires a datum or type path receiver, received {value}"),
                        ));
                    }
                };
                let value = state
                    .initial_value(&runtime_type, &name)
                    .cloned()
                    .unwrap_or(Value::Null);
                frames[frame_index].stack.push(value);
            }
'''
text = replace_once(text, old_load, new_load, "load/initial fields")

text = replace_once(
    text,
    "            Instruction::Add\n            | Instruction::Subtract\n            | Instruction::Multiply\n",
    "            Instruction::Add => {\n                let right = pop(&mut frames[frame_index].stack)\n                    .map_err(|message| execution_error(module, &frames, message))?;\n                let left = pop(&mut frames[frame_index].stack)\n                    .map_err(|message| execution_error(module, &frames, message))?;\n                let value = match (left, right) {\n                    (Value::Number(left), Value::Number(right)) => {\n                        Value::number(left.to_f32() + right.to_f32())\n                    }\n                    (Value::Text(left), Value::Text(right)) => {\n                        Value::text(format!(\"{left}{right}\"))\n                    }\n                    (left, right) => {\n                        return Err(execution_error(\n                            module,\n                            &frames,\n                            format!(\"addition requires two numbers or two text values, received {left} and {right}\"),\n                        ));\n                    }\n                };\n                frames[frame_index].stack.push(value);\n            }\n            Instruction::Subtract\n            | Instruction::Multiply\n            | Instruction::Power\n",
    "text add and power execution group",
)
text = replace_once(
    text,
    "        Instruction::Multiply => left * right,\n        Instruction::Divide => left / right,\n",
    "        Instruction::Multiply => left * right,\n        Instruction::Power => left.powf(right),\n        Instruction::Divide => left / right,\n",
    "execute power numeric",
)

text = text.replace(
    "                    TypePredicateKind::IsType => (1..=2).contains(&count),\n                    TypePredicateKind::IsLoc => count >= 1,\n                    _ => count == 1,\n",
    "                    TypePredicateKind::IsType | TypePredicateKind::IsPath => (1..=2).contains(&count),\n                    TypePredicateKind::IsLoc\n                    | TypePredicateKind::IsMovable\n                    | TypePredicateKind::IsTurf => count >= 1,\n                    _ => count == 1,\n",
    1,
)
text = replace_once(
    text,
    "                let result = type_predicate_builtin(kind, &arguments, &state.heap)\n",
    "                let result = type_predicate_builtin(kind, &arguments, state)\n",
    "type predicate state call",
)

old_pred_sig = '''fn type_predicate_builtin(
    kind: TypePredicateKind,
    arguments: &[Value],
    heap: &ValueHeap,
) -> Result<bool, String> {
'''
new_pred_sig = '''fn type_predicate_builtin(
    kind: TypePredicateKind,
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<bool, String> {
    let heap = &state.heap;
'''
text = replace_once(text, old_pred_sig, new_pred_sig, "type predicate signature")

old_mov = '''        TypePredicateKind::IsMovable => {
            let Value::Datum(datum) = value else {
                return Ok(false);
            };
            let type_path = heap
                .datum(*datum)
                .map_err(|error| error.to_string())?
                .type_path()
                .as_str();
            // `/obj` and `/mob` are conventional direct children of
            // `/atom/movable`, despite their path spelling not retaining that
            // parent segment. Their descendants are movable too.
            Ok(type_path == "/atom/movable"
                || type_path.starts_with("/atom/movable/")
                || type_path == "/obj"
                || type_path.starts_with("/obj/")
                || type_path == "/mob"
                || type_path.starts_with("/mob/"))
        }
        TypePredicateKind::IsTurf => {
            let Value::Datum(datum) = value else {
                return Ok(false);
            };
            let type_path = heap
                .datum(*datum)
                .map_err(|error| error.to_string())?
                .type_path()
                .as_str();
            Ok(type_path == "/turf" || type_path.starts_with("/turf/"))
        }
'''
new_mov = '''        TypePredicateKind::IsMovable => {
            let target = TypePath::parse("/atom/movable").expect("built-in movable path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
        TypePredicateKind::IsTurf => {
            let target = TypePath::parse("/turf").expect("built-in turf path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
'''
text = replace_once(text, old_mov, new_mov, "movable/turf predicates")

old_loc = '''        TypePredicateKind::IsLoc => Ok(arguments.iter().all(|value| {
            let Value::Datum(datum) = value else {
                return false;
            };
            let Ok(datum) = heap.datum(*datum) else {
                return false;
            };
            let type_path = datum.type_path().as_str();
            // The four concrete atom roots are conventionally spelled as
            // `/area`, `/turf`, `/obj`, and `/mob`, even though they inherit
            // `/atom` rather than retaining that segment in their paths.
            type_path == "/atom"
                || type_path.starts_with("/atom/")
                || type_path == "/area"
                || type_path.starts_with("/area/")
                || type_path == "/turf"
                || type_path.starts_with("/turf/")
                || type_path == "/obj"
                || type_path.starts_with("/obj/")
                || type_path == "/mob"
                || type_path.starts_with("/mob/")
        })),
'''
new_loc = '''        TypePredicateKind::IsLoc => {
            let target = TypePath::parse("/atom").expect("built-in atom path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
'''
text = replace_once(text, old_loc, new_loc, "isloc predicate")

old_ispath = '''        TypePredicateKind::IsPath => Ok(matches!(value, Value::TypePath(_))),
'''
new_ispath = '''        TypePredicateKind::IsPath => {
            let Value::TypePath(candidate) = value else {
                return Ok(false);
            };
            let Some(target) = arguments.get(1) else {
                return Ok(true);
            };
            let Value::TypePath(target) = target else {
                return Ok(false);
            };
            Ok(is_subtype(state, candidate, target))
        }
'''
text = replace_once(text, old_ispath, new_ispath, "ispath predicate")

old_istype_tail = '''            let target = target.as_str();
            let candidate = candidate.as_str();
            Ok(candidate == target
                || candidate
                    .strip_prefix(target)
                    .is_some_and(|suffix| suffix.starts_with('/')))
'''
new_istype_tail = '''            Ok(is_subtype(state, candidate, target))
'''
text = replace_once(text, old_istype_tail, new_istype_tail, "istype parent catalog")

# Add regression tests before the existing direction/icon test.
test_anchor = '''    #[test]
    fn direction_and_icon_builtins_cover_lifecycle_shapes() {
'''
tests = r'''    #[test]
    fn documented_native_builtins_cover_text_math_and_type_helpers() {
        let source = parse(
            "/proc/native(kind)\n\tvar/path = text2path(\"/datum/child\")\n\tif(!path)\n\t\treturn 0\n\treturn (2 ** 3 ** 2) + floor(1.9) + abs(-2) + findlasttext(\"/datum/child\", \"/\") + initial(kind.flag)\n",
        )
        .expect("native builtin source should parse");
        let module = compile_module(&source.definitions).expect("native builtins should compile");
        let mut state = ExecutionState::new();
        let base = TypePath::parse("/datum/base").unwrap();
        let child = TypePath::parse("/datum/child").unwrap();
        state.set_type_paths([base.clone(), child.clone()]);
        state.set_type_parents(BTreeMap::from([
            (base.clone(), Some(TypePath::parse("/datum").unwrap())),
            (child.clone(), Some(base.clone())),
        ]));
        state.set_initial_values(BTreeMap::from([(
            child.clone(),
            BTreeMap::from([(field("flag"), Value::number(7.0))]),
        )]));
        let result = execute_module_in_state(
            &module,
            module.procedure_id("/proc/native").unwrap(),
            &[Value::TypePath(child)],
            &mut state,
        )
        .expect("native builtin procedure should execute");
        // 2 ** (3 ** 2) = 512; floor=1; abs=2; final slash is byte 7; initial=7.
        assert_eq!(result, Value::number(529.0));
    }

    #[test]
    fn type_predicates_follow_runtime_parent_catalog_not_path_spelling() {
        let source = parse(
            "/proc/check(value)\n\treturn istype(value, /atom/movable) && ismovable(value)\n",
        )
        .expect("predicate source should parse");
        let module = compile_module(&source.definitions).expect("predicate source should compile");
        let mut state = ExecutionState::new();
        let obj = TypePath::parse("/obj/item").unwrap();
        state.set_type_parents(BTreeMap::from([
            (obj.clone(), Some(TypePath::parse("/obj").unwrap())),
            (
                TypePath::parse("/obj").unwrap(),
                Some(TypePath::parse("/atom/movable").unwrap()),
            ),
            (
                TypePath::parse("/atom/movable").unwrap(),
                Some(TypePath::parse("/atom").unwrap()),
            ),
        ]));
        let datum = state.heap_mut().allocate_datum(obj);
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/check").unwrap(),
                &[Value::Datum(datum)],
                &mut state,
            ),
            Ok(Value::number(1.0))
        );
    }

'''
text = replace_once(text, test_anchor, tests + test_anchor, "native builtin tests")
p.write_text(text)

# Runtime image: retain project root, transfer parent/default metadata, and
# materialize built-in world values used by platform branches.
p = Path("crates/dm-runtime/src/lib.rs")
text = p.read_text()
text = replace_once(
    text,
    "use std::path::Path;\n",
    "use std::path::{Path, PathBuf};\n",
    "runtime path imports",
)
text = replace_once(
    text,
    "    diagnostics: Vec<RuntimeInitializerDiagnostic>,\n    stats: RuntimeImageStats,\n}\n",
    "    diagnostics: Vec<RuntimeInitializerDiagnostic>,\n    project_root: PathBuf,\n    stats: RuntimeImageStats,\n}\n",
    "runtime image project root field",
)
text = replace_once(
    text,
    "            diagnostics: Vec::new(),\n            stats: RuntimeImageStats {\n",
    "            diagnostics: Vec::new(),\n            project_root: compilation.project().root_directory.clone(),\n            stats: RuntimeImageStats {\n",
    "runtime image project root init",
)

world_helper = r'''
fn materialize_builtin_world_defaults(
    heap: &mut ValueHeap,
    datum: DatumId,
    type_path: &TypePath,
) -> Result<(), ValueError> {
    if type_path.as_str() != "/world" {
        return Ok(());
    }
    let system_type = if cfg!(windows) { "MS Windows" } else { "UNIX" };
    let defaults: &[(&str, Value)] = &[
        ("system_type", Value::text(system_type)),
        ("icon_size", Value::number(32.0)),
        ("tick_lag", Value::number(1.0)),
        ("fps", Value::number(10.0)),
        ("timezone", Value::number(0.0)),
        ("cpu", Value::number(0.0)),
        ("time", Value::number(0.0)),
        ("timeofday", Value::number(0.0)),
        ("realtime", Value::number(0.0)),
    ];
    for (name, value) in defaults {
        let name = FieldName::parse(name).expect("built-in world field is valid");
        if heap.datum_field(datum, &name).is_err() {
            heap.set_datum_field(datum, name, value.clone())?;
        }
    }
    Ok(())
}

fn builtin_initial_fields(path: &TypePath) -> BTreeMap<FieldName, Value> {
    let mut fields = BTreeMap::new();
    let mut insert = |name: &str, value: Value| {
        fields.insert(
            FieldName::parse(name).expect("built-in initial field name is valid"),
            value,
        );
    };
    match path.as_str() {
        "/datum" => insert("tag", Value::Null),
        "/atom" => {
            for (name, value) in [
                ("alpha", Value::number(255.0)),
                ("appearance_flags", Value::number(0.0)),
                ("blend_mode", Value::number(0.0)),
                ("color", Value::Null),
                ("density", Value::number(0.0)),
                ("dir", Value::number(2.0)),
                ("icon", Value::Null),
                ("icon_state", Value::Null),
                ("invisibility", Value::number(0.0)),
                ("layer", Value::number(1.0)),
                ("loc", Value::Null),
                ("opacity", Value::number(0.0)),
                ("overlays", Value::Null),
                ("plane", Value::number(0.0)),
                ("underlays", Value::Null),
                ("x", Value::number(0.0)),
                ("y", Value::number(0.0)),
                ("z", Value::number(0.0)),
            ] {
                insert(name, value);
            }
        }
        "/atom/movable" => {
            for (name, value) in [
                ("animate_movement", Value::number(0.0)),
                ("bound_height", Value::number(32.0)),
                ("bound_width", Value::number(32.0)),
                ("bound_x", Value::number(0.0)),
                ("bound_y", Value::number(0.0)),
                ("glide_size", Value::number(0.0)),
                ("pixel_x", Value::number(0.0)),
                ("pixel_y", Value::number(0.0)),
                ("screen_loc", Value::Null),
                ("step_size", Value::number(32.0)),
            ] {
                insert(name, value);
            }
        }
        "/world" => {
            insert(
                "system_type",
                Value::text(if cfg!(windows) { "MS Windows" } else { "UNIX" }),
            );
            insert("icon_size", Value::number(32.0));
            insert("tick_lag", Value::number(1.0));
            insert("fps", Value::number(10.0));
        }
        _ => {}
    }
    fields
}

'''
text = replace_once(
    text,
    "impl RuntimeType {\n",
    world_helper + "impl RuntimeType {\n",
    "runtime builtin metadata helpers",
)

old_take = '''    pub fn take_execution_state(&mut self) -> ExecutionState {
        let mut state = ExecutionState::from_heap(std::mem::take(&mut self.heap));
        state.set_shared_type_paths(Arc::clone(&self.type_paths));
        for field in self.binding_index.globals.values() {
'''
new_take = '''    pub fn take_execution_state(&mut self) -> ExecutionState {
        let mut state = ExecutionState::from_heap(std::mem::take(&mut self.heap));
        state.set_shared_type_paths(Arc::clone(&self.type_paths));
        state.set_type_parents(
            self.types
                .iter()
                .map(|(path, runtime_type)| (path.clone(), runtime_type.parent.clone()))
                .collect(),
        );
        let mut initial_values = BTreeMap::new();
        for path in self.types.keys() {
            let mut hierarchy = Vec::new();
            let mut current = Some(path.clone());
            let mut visited = BTreeSet::new();
            while let Some(candidate) = current {
                if !visited.insert(candidate.clone()) {
                    break;
                }
                let Some(runtime_type) = self.types.get(&candidate) else {
                    break;
                };
                hierarchy.push(candidate.clone());
                current.clone_from(&runtime_type.parent);
            }
            hierarchy.reverse();
            let mut values = BTreeMap::new();
            for ancestor in hierarchy {
                values.extend(builtin_initial_fields(&ancestor));
                if let Some(runtime_type) = self.types.get(&ancestor) {
                    values.extend(
                        runtime_type
                            .defaults
                            .fields()
                            .map(|(field, value)| (field.clone(), value.clone())),
                    );
                }
            }
            values.insert(
                FieldName::parse("type").expect("built-in type field is valid"),
                Value::TypePath(path.clone()),
            );
            values.insert(
                FieldName::parse("parent_type").expect("built-in parent_type field is valid"),
                self.types
                    .get(path)
                    .and_then(|runtime_type| runtime_type.parent.clone())
                    .map_or(Value::Null, Value::TypePath),
            );
            initial_values.insert(path.clone(), values);
        }
        state.set_initial_values(initial_values);
        state.set_project_root(self.project_root.clone());
        for field in self.binding_index.globals.values() {
'''
text = replace_once(text, old_take, new_take, "runtime state metadata transfer")

text = replace_once(
    text,
    "        materialize_builtin_atom_defaults(&mut self.heap, datum, is_atom, is_movable)?;\n        self.stats.datums_allocated += 1;\n",
    "        materialize_builtin_atom_defaults(&mut self.heap, datum, is_atom, is_movable)?;\n        materialize_builtin_world_defaults(&mut self.heap, datum, type_path)?;\n        self.stats.datums_allocated += 1;\n",
    "world defaults allocation",
)

runtime_test_anchor = '''    #[test]
    fn execution_states_share_the_image_type_catalog() {
'''
runtime_test = r'''    #[test]
    fn execution_state_carries_initial_parent_and_project_metadata() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/value = 7\n/datum/base/child\n",
        );
        let mut image = fixture.image();
        let state = image.take_execution_state();
        let child = type_path("/datum/base/child");
        assert_eq!(state.type_parent(&child), Some(&type_path("/datum/base")));
        assert_eq!(
            state.initial_value(&child, &field("value")),
            Some(&Value::number(7.0))
        );
        assert_eq!(state.project_root(), Some(fixture.root.as_path()));
    }

'''
text = replace_once(text, runtime_test_anchor, runtime_test + runtime_test_anchor, "runtime metadata test")
p.write_text(text)

# Semantic standard fields: parent_type on every datum and core /world vars.
p = Path("crates/dm-semantics/src/lib.rs")
text = p.read_text()
text = replace_once(
    text,
    '        "/datum" => &["tag", "type"],\n',
    '        "/datum" => &["tag", "type", "parent_type"],\n        "/world" => &[\n            "system_type",\n            "icon_size",\n            "tick_lag",\n            "fps",\n            "timezone",\n            "cpu",\n            "time",\n            "timeofday",\n            "realtime",\n            "maxx",\n            "maxy",\n            "maxz",\n        ],\n',
    "standard datum/world fields",
)
p.write_text(text)
