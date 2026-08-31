//! Text builtins: `replacetext` / `replacetextEx` (literal and `/regex`),
//! `copytext`, `text()` coercion, and their position/bounds helpers.
//!
//! Split out of `value_ops`: DM's string-manipulation builtins and the
//! character/byte index math they share.

use crate::ExecutionState;
use crate::builtins;
use crate::bytecode::Module;
use crate::execute_module_in_context;
use dm_value::{DatumId, FieldName, Value, ValueHeap};

use super::{ExecutionContext, dynamic_call_target};

pub(crate) fn replace_text_builtin(
    arguments: &[Value],
    exact: bool,
    character_indices: bool,
    heap: &ValueHeap,
) -> Result<Value, String> {
    // BYOND's haystack and needle parameters are text-typed: a non-text
    // haystack returns null, while a non-text needle is the empty string.
    // Replacement is deliberately different and uses normal DM stringification
    // (for example the numeric constant 90 becomes "90").
    let Value::Text(source) = &arguments[0] else {
        return Ok(Value::Null);
    };
    let source = source.to_string();
    let needle = match &arguments[1] {
        Value::Text(text) => text.to_string(),
        _ => String::new(),
    };
    let replacement = stringify_dm_value(&arguments[2], heap)?;
    if needle.is_empty() {
        if replacement.is_empty() {
            return Ok(Value::text(source));
        }
        return replace_empty_needle(&source, &replacement, arguments, character_indices)
            .map(Value::text);
    }

    let (start, end) = replacement_bounds(&source, arguments, character_indices)?;
    let prefix = &source[..start];
    let target = &source[start..end];
    let suffix = &source[end..];
    let replaced = if exact {
        target.replace(&needle, &replacement)
    } else {
        replace_text_ascii_insensitive(target, &needle, &replacement)
    };
    Ok(Value::text(format!("{prefix}{replaced}{suffix}")))
}

pub(crate) fn stringify_dm_value(value: &Value, heap: &ValueHeap) -> Result<String, String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::Text(text) | Value::File(text) => Ok(text.to_string()),
        Value::Number(number) => Ok(number.to_f32().to_string()),
        Value::TypePath(path) => Ok(path.to_string()),
        Value::ModifiedTypePath(path) => Ok(path.base().to_string()),
        Value::Datum(datum) => {
            let datum = heap.datum(*datum).map_err(|error| error.to_string())?;
            let name = FieldName::parse("name").expect("built-in datum name is valid");
            if let Ok(Value::Text(name)) = datum.field(&name) {
                Ok(name.to_string())
            } else {
                Ok(datum.type_path().to_string())
            }
        }
        Value::List(_) => Ok("/list".to_owned()),
    }
}

pub(crate) fn replace_empty_needle(
    source: &str,
    replacement: &str,
    arguments: &[Value],
    character_indices: bool,
) -> Result<String, String> {
    let logical_length = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let limit = i64::try_from(logical_length)
        .unwrap_or(i64::MAX - 1)
        .saturating_add(1);
    let mut start = signed_text_index(arguments.get(3), 1)?;
    if start == 0 {
        return Ok(source.to_owned());
    }
    if start < 0 {
        start = limit.saturating_add(start).max(1);
    }
    let mut end = signed_text_index(arguments.get(4), 0)?;
    if end <= 0 {
        end = limit.saturating_add(end).max(start);
    }
    let mut start = usize::try_from(start.clamp(1, limit)).unwrap_or(usize::MAX);
    let mut end = usize::try_from(end.clamp(1, limit)).unwrap_or(usize::MAX);
    if start == 1 {
        start = 2;
    }
    end = end.min(logical_length);

    let mut output = String::with_capacity(source.len().saturating_add(replacement.len()));
    for (zero_based, character) in source.chars().enumerate() {
        output.push(character);
        let position = zero_based.saturating_add(1);
        if position >= start.saturating_sub(1) && position < end {
            output.push_str(replacement);
        }
    }
    Ok(output)
}

pub(crate) fn replace_text_regex(
    module: &Module,
    state: &mut ExecutionState,
    regex: DatumId,
    arguments: &[Value],
    character_indices: bool,
    caller_context: &ExecutionContext,
) -> Result<Value, String> {
    let Value::Text(source) = &arguments[0] else {
        return Ok(Value::Null);
    };
    let source = source.to_string();
    let field = |name| FieldName::parse(name).expect("regex field is valid");
    let pattern = state
        .heap()
        .datum_field(regex, &field("_dream64_pattern"))
        .map_err(|error| error.to_string())?
        .clone();
    let pattern = builtin_text(&pattern, &state.heap, "regex pattern")?;
    let flags = state
        .heap()
        .datum_field(regex, &field("flags"))
        .ok()
        .and_then(|value| match value {
            Value::Text(text) => Some(text.as_ref()),
            _ => None,
        })
        .unwrap_or("")
        .to_owned();
    let global = flags.contains('g');
    let (start, end) = replacement_bounds(&source, arguments, character_indices)?;
    let prefix = source[..start].to_owned();
    let suffix = source[end..].to_owned();
    let mut target = source[start..end].to_owned();
    let replacement_proc = matches!(arguments[2], Value::TypePath(_)).then(|| {
        dynamic_call_target(
            module,
            state,
            &Value::Null,
            &arguments[2],
            caller_context,
            true,
        )
    });
    let replacement_text = if replacement_proc.is_none() {
        Some(builtin_text(
            &arguments[2],
            &state.heap,
            "replacetext replacement",
        )?)
    } else {
        None
    };
    let replacement_proc = replacement_proc.transpose()?;

    let mut cursor = 0;
    loop {
        let Some((begin, finish, captures)) =
            builtins::regex_search(&pattern, &flags, &target, cursor, target.len())?
        else {
            break;
        };
        let replacement = if let Some((procedure, context)) = &replacement_proc {
            let mut callback_arguments = Vec::with_capacity(captures.len() + 1);
            callback_arguments.push(Value::text(&target[begin..finish]));
            callback_arguments.extend(
                captures
                    .into_iter()
                    .map(|capture| capture.map_or(Value::Null, Value::text)),
            );
            let value =
                execute_module_in_context(module, *procedure, &callback_arguments, state, context)
                    .map_err(|error| error.to_string())?;
            match value {
                Value::Null => String::new(),
                Value::Text(text) => text.to_string(),
                Value::Number(number) => number.to_f32().to_string(),
                Value::TypePath(path) => path.to_string(),
                value => format!("{value}"),
            }
        } else {
            replacement_text.clone().unwrap_or_default()
        };
        target.replace_range(begin..finish, &replacement);
        cursor = begin.saturating_add(replacement.len().max(1));
        if !global {
            break;
        }
    }
    Ok(Value::text(format!("{prefix}{target}{suffix}")))
}

pub(crate) fn copy_text_builtin(
    arguments: &[Value],
    character_indices: bool,
    heap: &ValueHeap,
) -> Result<String, String> {
    let source = builtin_text(&arguments[0], heap, "copytext text")?;
    let logical_length = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let start = signed_text_index(arguments.get(1), 1)?;
    let end = signed_text_index(arguments.get(2), 0)?;
    let start = resolve_text_position(start, logical_length);
    let end = if end == 0 {
        logical_length.saturating_add(1)
    } else {
        resolve_text_position(end, logical_length)
    };
    if end <= start {
        return Ok(String::new());
    }
    let start = start.saturating_sub(1);
    let end = end.saturating_sub(1);
    let (start, end) = if character_indices {
        (
            character_offset(&source, start),
            character_offset(&source, end),
        )
    } else {
        (
            previous_char_boundary(&source, start),
            previous_char_boundary(&source, end),
        )
    };
    Ok(source[start..end].to_owned())
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "DM text positions are integralized from binary32 at the language boundary"
)]
pub(crate) fn signed_text_index(value: Option<&Value>, default: i64) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => {
            let number = number.to_f32();
            if !number.is_finite() {
                return Ok(default);
            }
            Ok(number.trunc() as i64)
        }
        Some(value) => Err(format!(
            "copytext bounds require a number, received {value}"
        )),
    }
}

pub(crate) fn resolve_text_position(position: i64, logical_length: usize) -> usize {
    let limit = i64::try_from(logical_length)
        .unwrap_or(i64::MAX - 1)
        .saturating_add(1);
    let position = if position < 0 {
        limit.saturating_add(position)
    } else {
        position
    };
    usize::try_from(position.clamp(1, limit)).unwrap_or(usize::MAX)
}

pub(crate) fn builtin_text(
    value: &Value,
    heap: &ValueHeap,
    context: &str,
) -> Result<String, String> {
    match value {
        Value::Text(text) => Ok(String::from(text.as_ref())),
        Value::Datum(datum)
            if heap
                .datum(*datum)
                .map_err(|error| error.to_string())?
                .type_path()
                .to_string()
                == "/regex" =>
        {
            Err(format!("{context} regex matching is not yet supported"))
        }
        _ => Err(format!("{context} requires text, received {value}")),
    }
}

pub(crate) fn replacement_bounds(
    source: &str,
    arguments: &[Value],
    character_indices: bool,
) -> Result<(usize, usize), String> {
    let index_limit = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let start = optional_text_index(arguments.get(3), 1)?;
    let end = optional_text_index(arguments.get(4), 0)?;
    // BYOND text positions are 1-based and the end is exclusive; zero end
    // extends through the whole remaining text.
    let start = start.clamp(1, index_limit.saturating_add(1));
    let end = if end == 0 {
        index_limit.saturating_add(1)
    } else {
        end.clamp(start, index_limit.saturating_add(1))
    };
    if character_indices {
        Ok((
            character_offset(source, start.saturating_sub(1)),
            character_offset(source, end.saturating_sub(1)),
        ))
    } else {
        // Legacy DM indices count UTF-8 bytes. Clamp inward to valid Rust
        // boundaries instead of manufacturing invalid text slices.
        Ok((
            previous_char_boundary(source, start.saturating_sub(1)),
            previous_char_boundary(source, end.saturating_sub(1)),
        ))
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "DM text positions are non-negative integral binary32 values"
)]
pub(crate) fn optional_text_index(value: Option<&Value>, default: usize) -> Result<usize, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => Ok(number.to_f32().max(0.0) as usize),
        Some(value) => Err(format!(
            "replacetext bounds require a number, received {value}"
        )),
    }
}

pub(crate) fn character_offset(text: &str, character_index: usize) -> usize {
    text.char_indices()
        .nth(character_index)
        .map_or(text.len(), |(byte, _)| byte)
}

pub(crate) fn previous_char_boundary(text: &str, mut byte_index: usize) -> usize {
    byte_index = byte_index.min(text.len());
    while byte_index > 0 && !text.is_char_boundary(byte_index) {
        byte_index -= 1;
    }
    byte_index
}

pub(crate) fn replace_text_ascii_insensitive(
    target: &str,
    needle: &str,
    replacement: &str,
) -> String {
    if !needle.is_ascii() {
        // DM's Unicode case folding is more involved than Rust's simple
        // lowercase mapping. Preserve deterministic exact text for the rare
        // non-ASCII fallback rather than corrupting byte offsets.
        return target.replace(needle, replacement);
    }
    let needle_lower = needle.to_ascii_lowercase();
    let bytes = target.as_bytes();
    let mut output = String::with_capacity(target.len());
    let mut cursor = 0;
    while cursor < target.len() {
        let remaining = &target[cursor..];
        if remaining.len() >= needle.len()
            && remaining.as_bytes()[..needle.len()]
                .iter()
                .map(u8::to_ascii_lowercase)
                .eq(needle_lower.bytes())
        {
            output.push_str(replacement);
            cursor += needle.len();
        } else {
            let width = char::from(bytes[cursor]).len_utf8();
            output.push_str(&target[cursor..cursor + width]);
            cursor += width;
        }
    }
    output
}
