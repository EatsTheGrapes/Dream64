//! Text intrinsics: `length`/`ascii`/numeric coercion, case folding, search
//! (`findtext`, `spantext`), the `/regex` native search cluster with its
//! pattern cache, plus split/join/splice and the `splittext` allocation tests.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use dm_value::{DatumId, FieldName, Value};

use super::{ExecutionState, number, runtime_text, strict_text, truthy, value_text};

pub(super) fn length_char(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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

pub(super) fn ascii2text(arguments: &[Value]) -> Result<Value, String> {
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

pub(super) fn text2ascii(
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

pub(super) fn text2num(arguments: &[Value], _state: &ExecutionState) -> Result<Value, String> {
    let text = match &arguments[0] {
        Value::Number(number) => return Ok(Value::Number(*number)),
        Value::Null => return Ok(Value::Null),
        Value::Text(text) => text.as_ref(),
        _ => return Ok(Value::Null),
    };
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

pub(super) fn text2path(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    // BYOND 516 returns null for every non-text input, including an already
    // resolved type path. Only the textual spelling participates in lookup.
    let Value::Text(text) = &arguments[0] else {
        return Ok(Value::Null);
    };
    Ok(state
        .type_paths
        .get(text.as_ref())
        .cloned()
        .map_or(Value::Null, Value::TypePath))
}

pub(super) fn numeric_classifier(
    arguments: &[Value],
    predicate: impl FnOnce(f32) -> bool,
) -> Result<Value, String> {
    Ok(Value::number(f32::from(match &arguments[0] {
        Value::Number(number) => predicate(number.to_f32()),
        _ => false,
    })))
}

pub(super) fn cmptext(
    arguments: &[Value],
    state: &ExecutionState,
    exact: bool,
) -> Result<Value, String> {
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

pub(super) fn findtext(
    arguments: &[Value],
    state: &mut ExecutionState,
    exact: bool,
    character_indices: bool,
    reverse: bool,
) -> Result<Value, String> {
    // BYOND text searches accept null as an empty text value. This is
    // observable when `file2text()` returns null for a directory entry from
    // `flist()`: map readers probe the result with `findtext()` before their
    // regex loop and must receive a normal no-match rather than a runtime.
    let haystack = match &arguments[0] {
        Value::Null => String::new(),
        Value::Text(text) | Value::File(text) => text.to_string(),
        // BYOND's text-search family returns a normal no-match for a
        // non-text haystack instead of raising a runtime. Monkestation's
        // immune system relies on this when an older call site passes its
        // `/datum/blood_type` singleton directly to `findtext()`.
        _ => return Ok(Value::number(0.0)),
    };
    if let Value::Datum(regex) = arguments[1] {
        let start = signed_position(arguments.get(2), 1)?.max(1) as usize;
        let end = signed_position(arguments.get(3), 0)?;
        let end = if end <= 0 {
            haystack.len() + 1
        } else {
            end as usize
        };
        let haystack: Arc<str> = Arc::from(haystack);
        return regex_find(regex, &haystack, start, end, false, false, state);
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

pub(crate) fn is_regex_datum(datum: DatumId, state: &ExecutionState) -> bool {
    state
        .heap()
        .datum(datum)
        .is_ok_and(|value| value.type_path().as_str() == "/regex")
}

pub(crate) fn execute_regex_method(
    datum: DatumId,
    method: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if method != "Find" || arguments.is_empty() || arguments.len() > 3 {
        return Err(format!("unknown or invalid /regex procedure {method:?}"));
    }
    // `/regex.Find()` applies the same null-to-empty text coercion as the
    // global text-search procedures. In particular, this lets a parsed-map
    // reader finish cleanly after `file2text()` rejected a directory.
    let haystack: Arc<str> = match &arguments[0] {
        Value::Null => Arc::from(""),
        Value::Text(text) | Value::File(text) => Arc::clone(text),
        _ => {
            return Err(format!(
                "regex.Find haystack requires text, received {}",
                runtime_text(&arguments[0], state, "regex.Find haystack")?
            ));
        }
    };
    let supplied_start = arguments
        .get(1)
        .is_some_and(|value| !matches!(value, Value::Null));
    let start = signed_position(arguments.get(1), 1)?.max(1) as usize;
    let end = signed_position(arguments.get(2), 0)?;
    let end = if end <= 0 {
        haystack.len() + 1
    } else {
        end as usize
    };
    regex_find(datum, &haystack, start, end, true, !supplied_start, state)
}

fn regex_find(
    datum: DatumId,
    haystack: &Arc<str>,
    requested_start: usize,
    requested_end: usize,
    method_call: bool,
    use_global_cursor: bool,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let field = |name| FieldName::parse(name).expect("regex field is valid");
    let pattern = state
        .heap()
        .datum_field(datum, &field("_dream64_pattern"))
        .map_err(|error| error.to_string())?
        .clone();
    let pattern = strict_text(&pattern, state, "regex pattern")?;
    let flags = state
        .heap()
        .datum_field(datum, &field("flags"))
        .ok()
        .and_then(value_text)
        .unwrap_or("")
        .to_owned();
    let global = flags.contains('g');
    let previous = state
        .heap()
        .datum_field(datum, &field("_dream64_haystack"))
        .ok()
        .and_then(value_text);
    let cursor = state
        .heap()
        .datum_field(datum, &field("_dream64_cursor"))
        .ok()
        .and_then(Value::as_number)
        .unwrap_or(0.0) as usize;
    let start = if global && use_global_cursor && previous == Some(haystack) && cursor > 0 {
        cursor
    } else {
        requested_start.saturating_sub(1)
    };
    let end = requested_end.saturating_sub(1).min(haystack.len());
    let found = regex_search(&pattern, &flags, haystack, start.min(end), end)?;
    let Some((begin, finish, captures)) = found else {
        if global {
            for (name, value) in [
                ("next", Value::Null),
                ("_dream64_cursor", Value::number(0.0)),
            ] {
                state
                    .heap_mut()
                    .set_datum_field(datum, field(name), value)
                    .map_err(|e| e.to_string())?;
            }
        }
        if method_call {
            state
                .heap_mut()
                .set_datum_field(datum, field("text"), Value::Text(Arc::clone(haystack)))
                .map_err(|e| e.to_string())?;
        }
        return Ok(Value::number(0.0));
    };
    let groups = state.heap_mut().allocate_list();
    for capture in captures {
        state
            .heap_mut()
            .list_mut(groups)
            .map_err(|error| error.to_string())?
            .add(capture.map_or(Value::Null, Value::text));
    }
    let next = if finish > begin {
        finish
    } else {
        finish.saturating_add(1)
    };
    let mut fields = vec![
        ("match", Value::text(&haystack[begin..finish])),
        ("index", Value::number((begin + 1) as f32)),
        ("group", Value::List(groups)),
        ("_dream64_cursor", Value::number(next as f32)),
        ("_dream64_haystack", Value::Text(Arc::clone(haystack))),
    ];
    if global {
        fields.push(("next", Value::number((next + 1) as f32)));
    }
    if method_call {
        fields.push(("text", Value::Text(Arc::clone(haystack))));
    }
    for (name, value) in fields {
        state
            .heap_mut()
            .set_datum_field(datum, field(name), value)
            .map_err(|e| e.to_string())?;
    }
    Ok(Value::number((begin + 1) as f32))
}

pub(crate) fn regex_search(
    pattern: &str,
    flags: &str,
    haystack: &str,
    start: usize,
    end: usize,
) -> Result<Option<(usize, usize, Vec<Option<String>>)>, String> {
    let pattern = translate_byond_regex_pattern(pattern);
    let case_insensitive = flags.contains('i');
    let multi_line = flags.contains('m');
    type RegexCache = HashMap<(String, bool, bool), Arc<fancy_regex::Regex>>;
    static REGEX_CACHE: OnceLock<Mutex<RegexCache>> = OnceLock::new();
    let key = (pattern.clone(), case_insensitive, multi_line);
    let cache = REGEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let regex = cache
        .lock()
        .map_err(|_| "regex cache lock is poisoned".to_owned())?
        .get(&key)
        .cloned();
    let regex = if let Some(regex) = regex {
        regex
    } else {
        let mut builder = fancy_regex::RegexBuilder::new(&pattern);
        builder
            .case_insensitive(case_insensitive)
            .multi_line(multi_line);
        let regex = Arc::new(
            builder
                .build()
                .map_err(|error| format!("invalid regex {pattern:?}: {error}"))?,
        );
        cache
            .lock()
            .map_err(|_| "regex cache lock is poisoned".to_owned())?
            .insert(key, Arc::clone(&regex));
        regex
    };
    let captures = regex
        .captures_from_pos(haystack, start)
        .map_err(|error| format!("regex match failed for {pattern:?}: {error}"))?;
    let Some(captures) = captures else {
        return Ok(None);
    };
    let Some(whole) = captures.get(0) else {
        return Ok(None);
    };
    if whole.end() > end {
        return Ok(None);
    }
    let groups = (1..captures.len())
        .map(|index| {
            captures
                .get(index)
                .map(|capture| capture.as_str().to_owned())
        })
        .collect();
    Ok(Some((whole.start(), whole.end(), groups)))
}

fn translate_byond_regex_pattern(pattern: &str) -> String {
    let mut translated = String::with_capacity(pattern.len());
    let mut characters = pattern.chars().peekable();
    let mut in_character_class = false;
    while let Some(character) = characters.next() {
        match character {
            '[' => {
                in_character_class = true;
                translated.push(character);
            }
            ']' => {
                in_character_class = false;
                translated.push(character);
            }
            '\\' if characters.peek() == Some(&'l') => {
                characters.next();
                if in_character_class {
                    translated.push_str("A-Za-z");
                } else {
                    translated.push_str("[A-Za-z]");
                }
            }
            _ => translated.push(character),
        }
    }
    translated
}

pub(super) fn splittext(
    arguments: &[Value],
    state: &mut ExecutionState,
    character_indices: bool,
) -> Result<Value, String> {
    if matches!(arguments[0], Value::Null) {
        return Ok(Value::List(state.heap.allocate_list()));
    }
    let text = strict_text(&arguments[0], state, "splittext text")?;
    let start = signed_position(arguments.get(2), 1)?;
    let end = signed_position(arguments.get(3), 0)?;
    let include_delimiters = arguments.get(4).is_some_and(truthy);
    let (region_start, region_end, _) = text_region(&text, start, end, character_indices);
    let target = &text[region_start..region_end];
    let list = state.heap.allocate_list();
    let mut output = Vec::new();
    if let Value::Datum(regex) = arguments[1] {
        if !is_regex_datum(regex, state) {
            return Ok(Value::List(list));
        }
        let field = |name| FieldName::parse(name).expect("regex field is valid");
        let pattern = state
            .heap()
            .datum_field(regex, &field("_dream64_pattern"))
            .map_err(|error| error.to_string())?
            .clone();
        let pattern = strict_text(&pattern, state, "splittext regex delimiter")?;
        let flags = state
            .heap()
            .datum_field(regex, &field("flags"))
            .ok()
            .and_then(value_text)
            .unwrap_or("")
            .to_owned();
        let mut segment_start = region_start;
        let mut search_start = region_start;
        while search_start <= region_end {
            let Some((found, finish, captures)) =
                regex_search(&pattern, &flags, &text, search_start, region_end)?
            else {
                break;
            };
            output.push(text[segment_start..found].to_owned());
            if include_delimiters {
                output.push(text[found..finish].to_owned());
            } else {
                output.extend(captures.into_iter().flatten());
            }
            segment_start = finish;
            search_start = if finish > found {
                finish
            } else {
                finish
                    + text[finish..region_end]
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or(1)
            };
        }
        output.push(text[segment_start..region_end].to_owned());
    } else {
        let delimiter = strict_text(&arguments[1], state, "splittext delimiter")?;
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
    }
    // BYOND applies Start/End only to the matching region. Text outside that
    // region remains attached to the first and last split elements.
    if output.is_empty() {
        output.push(text.clone());
    } else {
        output[0].insert_str(0, &text[..region_start]);
        output
            .last_mut()
            .expect("split output exists")
            .push_str(&text[region_end..]);
    }
    let entries = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    entries.reserve_positional(output.len());
    for item in output {
        entries.add(Value::text(item));
    }
    Ok(Value::List(list))
}

#[cfg(test)]
mod splittext_allocation_tests {
    use super::*;

    #[test]
    fn splittext_presizing_preserves_empty_and_trailing_lines() {
        let mut state = ExecutionState::new();
        let Value::List(lines) = splittext(
            &[
                Value::text("/turf/open,\n\ticon_state = \"floor\";\n\n"),
                Value::text("\n"),
            ],
            &mut state,
            false,
        )
        .unwrap() else {
            panic!("splittext should return a list")
        };
        let values = state
            .heap()
            .list(lines)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            vec![
                Value::text("/turf/open,"),
                Value::text("\ticon_state = \"floor\";"),
                Value::text(""),
                Value::text(""),
            ]
        );
    }
}

pub(super) fn jointext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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

pub(super) fn addtext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let mut output = String::new();
    for value in arguments {
        output.push_str(&strict_text(value, state, "addtext")?);
    }
    Ok(Value::text(output))
}

pub(super) fn spantext(
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

pub(super) fn splicetext(
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
