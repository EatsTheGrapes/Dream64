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
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::Component;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use dm_value::{DatumId, FieldName, ListId, TypePath, Value};

use super::{CompoundAssignmentOperator, ExecutionState, compare_values};

pub(super) fn standard_builtin_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "abs" | "ceil" | "floor" | "fract" | "trunc" | "sign" | "sqrt" | "sin" | "cos" | "tan"
        | "arcsin" | "arccos" | "length_char" | "lowertext" | "uppertext" | "trimtext"
        | "ascii2text" | "text2path" | "isinf" | "isnan" | "ckey" | "fexists" | "file2text"
        | "lentext" | "list2params" | "params2list" | "file" | "html_encode" | "html_decode"
        | "isfile" | "fdel" | "del" | "rand_seed" => (1, 1),
        "flist" => (0, 1),
        "fcopy_rsc" | "REGEX_QUOTE" | "REGEX_QUOTE_REPLACEMENT" => (1, 1),
        "browse" => (1, 2),
        "winset" => (2, 3),
        "winexists" => (2, 2),
        "alert" => (1, 6),
        "input" => (0, 4),
        "FLOOR" => (2, 2),
        "fcopy" => (2, 2),
        "text2file" => (2, 3),
        "json_decode" | "md5" => (0, 1),
        "json_encode" => (0, 2),
        "log" | "arctan" | "text2ascii" | "text2ascii_char" | "text2num" => (1, 2),
        "image" | "sort_list" | "qdel" | "typecacheof" | "icon" => (0, 5),
        "view" | "oview" | "viewers" | "hearers" => (1, 2),
        "step" => (2, 3),
        "sound" => (0, 7),
        "icon_states" => (1, 2),
        "newlist" => (0, usize::MAX),
        "min" | "max" => (0, usize::MAX),
        "clamp" | "lerp" => (3, 3),
        "cmptext" | "cmptextEx" | "sorttext" | "sorttextEx" | "sortText" | "addtext" => {
            (0, usize::MAX)
        }
        "text" => (1, usize::MAX),
        "num2text" => (1, 3),
        "time2text" => (1, 3),
        "rgb2num" => (1, 2),
        "rgb" => (3, 5),
        "gradient" => (2, usize::MAX),
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
        "get_dist" | "turn" | "astype" | "flick" | "output" => (2, 2),
        "values_cut_over" | "values_cut_under" => (2, 3),
        "values_dot" => (2, 2),
        "values_product" | "values_sum" => (1, 1),
        "_dream64_world_profile" => (1, 3),
        "_dream64_world_get_config" => (1, 2),
        "_dream64_world_set_config" => (3, 3),
        "_dream64_world_open_port" => (2, 2),
        _ => return None,
    })
}

pub(super) fn execute_standard_builtin(
    name: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match name {
        // Headless input has no interactive client. Preserve BYOND's supplied
        // default (the fourth positional argument), or null when absent.
        "input" => Ok(arguments.get(3).cloned().unwrap_or(Value::Null)),
        "text" => text_template(arguments, state),
        "newlist" => newlist_builtin(arguments, state),
        "qdel" => qdel_builtin(arguments, state),
        "del" => del_builtin(arguments, state),
        // `rand_seed()` resets the same per-world stream consumed by rand(),
        // prob(), pick(), roll(), and random-direction fallbacks.
        "rand_seed" => {
            let seed = number(&arguments[0], "rand_seed")?.trunc() as i64;
            state.random_state = seed as u64;
            Ok(Value::Null)
        }
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
        "min" => extrema_builtin(arguments, state, false),
        "max" => extrema_builtin(arguments, state, true),
        "length_char" => length_char(arguments, state),
        "lowertext" => text_map(arguments, state, str::to_lowercase),
        "uppertext" => text_map(arguments, state, str::to_uppercase),
        "trimtext" => text_map(arguments, state, |value| value.trim().to_owned()),
        "fcopy_rsc" => Ok(arguments.first().cloned().unwrap_or(Value::Null)),
        "REGEX_QUOTE" => regex_quote(arguments, state, false),
        "REGEX_QUOTE_REPLACEMENT" => regex_quote(arguments, state, true),
        "browse" => headless_browse(arguments, state),
        "winset" => headless_winset(arguments, state),
        "winexists" => headless_winexists(arguments, state),
        "alert" => headless_alert(arguments),
        "FLOOR" => floor_multiple(arguments),
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
        // `flick()` temporarily changes only the client-rendered icon state;
        // the atom's persistent `icon_state` is deliberately untouched.
        "flick" => Ok(Value::Null),
        "output" => resource_datum_builtin("/output", &["message", "control"], arguments, state),
        "values_cut_over" => values_cut(arguments, state, true),
        "values_cut_under" => values_cut(arguments, state, false),
        "values_dot" => values_dot(arguments, state),
        "values_product" => values_fold(arguments, state, true),
        "values_sum" => values_fold(arguments, state, false),
        "_dream64_world_profile" => world_profile(arguments, state),
        "_dream64_world_get_config" => world_get_config(arguments, state),
        "_dream64_world_set_config" => world_set_config(arguments, state),
        "_dream64_world_open_port" => world_open_port(arguments, state),
        "ckey" => ckey(arguments, state),
        "fexists" => fexists(arguments, state),
        "file2text" => file2text(arguments, state),
        "isfile" => Ok(Value::number(f32::from(matches!(
            arguments[0],
            Value::Text(_)
        )))),
        "fdel" => fdel(arguments, state),
        "flist" => flist(arguments, state),
        "fcopy" => fcopy(arguments, state),
        "text2file" => text2file(arguments, state),
        "html_encode" => html_encode(arguments, state),
        "html_decode" => html_decode(arguments, state),
        "rgb" => rgb_builtin(arguments),
        "rgb2num" => rgb2num_builtin(arguments, state),
        "gradient" => gradient_builtin(arguments, state),
        "time2text" => time2text_builtin(arguments, state),
        "view" => spatial_query(arguments, state, false, false),
        "oview" => spatial_query(arguments, state, false, true),
        "viewers" | "hearers" => spatial_query(arguments, state, true, false),
        "step" => step_builtin(arguments, state),
        "file" => match &arguments[0] {
            Value::Text(path) => Ok(Value::text(path.to_string())),
            value => Err(format!("file() requires a resource path, received {value}")),
        },
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
        "icon" => resource_datum_builtin(
            "/icon",
            &["icon", "icon_state", "dir", "frame", "moving"],
            arguments,
            state,
        ),
        "sound" => resource_datum_builtin(
            "/sound",
            &[
                "file",
                "repeat",
                "wait",
                "channel",
                "volume",
                "frequency",
                "pan",
            ],
            arguments,
            state,
        ),
        "icon_states" => {
            // DMI state discovery requires decoding the resource; retain a
            // deterministic empty list when no renderer/resource decoder is attached.
            Ok(Value::List(state.heap_mut().allocate_list()))
        }
        _ => Err(format!("unknown native DM builtin {name:?}")),
    }
}

pub(super) fn execute_external_call(
    library: &Value,
    function: &Value,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let library = strict_text(library, state, "external library")?;
    let function = strict_text(function, state, "external function")?;
    let filename = std::path::Path::new(&library)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(&library)
        .to_ascii_lowercase();
    if !matches!(
        filename.as_str(),
        "rust_g" | "rust_g.dll" | "librust_g.so" | "librust_g64.so"
    ) {
        return Err(format!(
            "external call {library}::{function} requires an installed host bridge"
        ));
    }
    match function.as_str() {
        "unix_timestamp" if arguments.is_empty() => {
            let seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| format!("unix_timestamp failed: {error}"))?
                .as_secs();
            Ok(Value::text(seconds.to_string()))
        }
        "file_write" | "file_append" if arguments.len() == 2 => {
            let text = strict_text(&arguments[0], state, function.as_str())?;
            let path = resolved_file_path(&arguments[1..], state, function.as_str())?;
            if function == "file_write" {
                fs::write(path, text).map_err(|error| format!("file_write failed: {error}"))?;
            } else {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|error| format!("file_append failed: {error}"))?;
                file.write_all(text.as_bytes())
                    .map_err(|error| format!("file_append failed: {error}"))?;
            }
            // rust-g's void file helpers yield BYOND null on success.
            Ok(Value::Null)
        }
        "toml_file_to_json" if arguments.len() == 1 => {
            let result = resolved_file_path(arguments, state, "toml_file_to_json")
                .and_then(|path| fs::read_to_string(path).map_err(|error| error.to_string()))
                .and_then(|source| parse_toml_document(&source));
            let (success, content) = match result {
                Ok(document) => (
                    true,
                    serde_json::to_string(&document).map_err(|error| error.to_string())?,
                ),
                Err(error) => (false, error),
            };
            Ok(Value::text(
                serde_json::json!({ "success": success, "content": content }).to_string(),
            ))
        }
        "time_reset" if arguments.len() == 1 => {
            let name = strict_text(&arguments[0], state, "time_reset")?;
            state.reset_external_timer(name);
            Ok(Value::Null)
        }
        "time_milliseconds" if arguments.len() == 1 => {
            let name = strict_text(&arguments[0], state, "time_milliseconds")?;
            Ok(Value::text(
                state.external_timer_milliseconds(&name).to_string(),
            ))
        }
        _ => Err(format!(
            "external call {library}::{function} requires an installed host bridge"
        )),
    }
}

fn parse_toml_document(source: &str) -> Result<serde_json::Value, String> {
    let mut root = serde_json::Map::new();
    let mut context = Vec::<String>::new();
    for (line_index, raw) in source.lines().enumerate() {
        let line = strip_toml_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("[[") && line.ends_with("]]") {
            context = parse_toml_key_path(&line[2..line.len() - 2])?;
            let (parent, leaf) = context
                .split_last()
                .ok_or_else(|| "empty array-table name".to_owned())?;
            let object = toml_object_at(&mut root, leaf)?;
            let array = object
                .entry(parent.clone())
                .or_insert_with(|| serde_json::Value::Array(Vec::new()))
                .as_array_mut()
                .ok_or_else(|| {
                    format!(
                        "line {}: array table conflicts with a value",
                        line_index + 1
                    )
                })?;
            array.push(serde_json::Value::Object(serde_json::Map::new()));
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            context = parse_toml_key_path(&line[1..line.len() - 1])?;
            toml_object_at(&mut root, &context)?;
            continue;
        }
        let (key, value) = split_toml_assignment(line)
            .ok_or_else(|| format!("line {}: expected key = value", line_index + 1))?;
        let mut path = context.clone();
        path.extend(parse_toml_key_path(key)?);
        let (leaf, parents) = path
            .split_last()
            .ok_or_else(|| format!("line {}: empty key", line_index + 1))?;
        let object = toml_object_at(&mut root, parents)?;
        if object
            .insert(leaf.clone(), parse_toml_value(value)?)
            .is_some()
        {
            return Err(format!("line {}: duplicate key {leaf}", line_index + 1));
        }
    }
    Ok(serde_json::Value::Object(root))
}

fn toml_object_at<'a>(
    root: &'a mut serde_json::Map<String, serde_json::Value>,
    path: &[String],
) -> Result<&'a mut serde_json::Map<String, serde_json::Value>, String> {
    let mut object = root;
    for segment in path {
        let mut value = object
            .entry(segment.clone())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        if let serde_json::Value::Array(array) = value {
            value = array
                .last_mut()
                .ok_or_else(|| format!("array table {segment} has no current entry"))?;
        }
        object = value
            .as_object_mut()
            .ok_or_else(|| format!("table {segment} conflicts with a value"))?;
    }
    Ok(object)
}

fn strip_toml_comment(line: &str) -> &str {
    let mut quote = None;
    let mut escaped = false;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some('"') && character == '\\' {
            escaped = true;
        } else if matches!(character, '"' | '\'') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
        } else if character == '#' && quote.is_none() {
            return &line[..index];
        }
    }
    line
}

fn split_toml_assignment(line: &str) -> Option<(&str, &str)> {
    let mut quote = None;
    let mut escaped = false;
    let mut depth = 0usize;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some('"') && character == '\\' {
            escaped = true;
        } else if matches!(character, '"' | '\'') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
        } else if quote.is_none() {
            match character {
                '[' | '{' => depth += 1,
                ']' | '}' => depth = depth.saturating_sub(1),
                '=' if depth == 0 => return Some((line[..index].trim(), line[index + 1..].trim())),
                _ => {}
            }
        }
    }
    None
}

fn parse_toml_key_path(source: &str) -> Result<Vec<String>, String> {
    split_toml_items(source, '.')
        .into_iter()
        .map(|part| parse_toml_key(part.trim()))
        .collect()
}

fn parse_toml_key(source: &str) -> Result<String, String> {
    if source.starts_with('"') {
        serde_json::from_str(source).map_err(|error| format!("invalid quoted key: {error}"))
    } else if source.starts_with('\'') && source.ends_with('\'') && source.len() >= 2 {
        Ok(source[1..source.len() - 1].to_owned())
    } else if !source.is_empty()
        && source
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
    {
        Ok(source.to_owned())
    } else {
        Err(format!("invalid TOML key {source:?}"))
    }
}

fn parse_toml_value(source: &str) -> Result<serde_json::Value, String> {
    let source = source.trim();
    if source.starts_with('"') {
        return serde_json::from_str::<String>(source)
            .map(serde_json::Value::String)
            .map_err(|error| format!("invalid string: {error}"));
    }
    if source.starts_with('\'') && source.ends_with('\'') && source.len() >= 2 {
        return Ok(serde_json::Value::String(
            source[1..source.len() - 1].to_owned(),
        ));
    }
    if source.starts_with('[') && source.ends_with(']') {
        let inner = &source[1..source.len() - 1];
        return split_toml_items(inner, ',')
            .into_iter()
            .filter(|item| !item.trim().is_empty())
            .map(parse_toml_value)
            .collect::<Result<Vec<_>, _>>()
            .map(serde_json::Value::Array);
    }
    match source {
        "true" => return Ok(serde_json::Value::Bool(true)),
        "false" => return Ok(serde_json::Value::Bool(false)),
        _ => {}
    }
    let number = source.replace('_', "");
    if let Ok(integer) = number.parse::<i64>() {
        return Ok(serde_json::Value::Number(integer.into()));
    }
    if let Ok(float) = number.parse::<f64>() {
        return serde_json::Number::from_f64(float)
            .map(serde_json::Value::Number)
            .ok_or_else(|| format!("non-finite TOML number {source}"));
    }
    Err(format!("unsupported TOML value {source:?}"))
}

fn split_toml_items(source: &str, delimiter: char) -> Vec<&str> {
    let mut items = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    let mut depth = 0usize;
    for (index, character) in source.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some('"') && character == '\\' {
            escaped = true;
            continue;
        }
        if matches!(character, '"' | '\'') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
        } else if quote.is_none() {
            match character {
                '[' | '{' => depth += 1,
                ']' | '}' => depth = depth.saturating_sub(1),
                _ if character == delimiter && depth == 0 => {
                    items.push(&source[start..index]);
                    start = index + character.len_utf8();
                }
                _ => {}
            }
        }
    }
    items.push(&source[start..]);
    items
}

fn world_profile(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let format = arguments
        .get(2)
        .and_then(value_text)
        .or_else(|| arguments.get(1).and_then(value_text));
    if format == Some("json") {
        return Ok(Value::text("[]"));
    }
    let columns: &[&str] = if arguments.get(1).and_then(value_text) == Some("sendmaps") {
        &["name", "value", "calls"]
    } else {
        &["name", "self", "total", "real", "over", "calls"]
    };
    let list = state.heap_mut().allocate_list();
    for column in columns {
        state
            .heap_mut()
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .add(Value::text(*column));
    }
    Ok(Value::List(list))
}

fn world_get_config(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let config_set = strict_text(&arguments[0], state, "world.GetConfig config set")?;
    let config_set = config_set.rsplit('/').next().unwrap_or(&config_set);
    match config_set {
        "env" => {
            let Some(name) = arguments.get(1).and_then(value_text) else {
                return Ok(Value::Null);
            };
            Ok(match state.environment_override(name) {
                Some(Some(value)) => value.clone(),
                Some(None) => Value::Null,
                None => std::env::var(name).map_or(Value::Null, Value::text),
            })
        }
        "ban" | "keyban" | "ipban" | "admin" => Ok(Value::List(state.heap_mut().allocate_list())),
        _ => Err(format!("unknown world configuration set {config_set:?}")),
    }
}

fn world_set_config(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let config_set = strict_text(&arguments[0], state, "world.SetConfig config set")?;
    let config_set = config_set.rsplit('/').next().unwrap_or(&config_set);
    match config_set {
        "env" => {
            let name = strict_text(&arguments[1], state, "world.SetConfig parameter")?;
            let value = value_text(&arguments[2]).map(Value::text);
            state.set_environment_override(name, value);
        }
        "ban" | "keyban" | "ipban" | "admin" => {}
        _ => return Err(format!("unknown world configuration set {config_set:?}")),
    }
    Ok(Value::Null)
}

fn value_text(value: &Value) -> Option<&str> {
    match value {
        Value::Text(text) => Some(text),
        _ => None,
    }
}

fn world_open_port(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let Value::Datum(world) = arguments[0] else {
        return Err("world.OpenPort requires a world datum receiver".to_owned());
    };
    let port = arguments[1]
        .as_number()
        .ok_or_else(|| "world.OpenPort requires a numeric port".to_owned())?;
    state
        .heap_mut()
        .set_datum_field(
            world,
            FieldName::parse("port").expect("built-in world field is valid"),
            Value::number(port),
        )
        .map_err(|error| error.to_string())?;
    Ok(Value::number(1.0))
}

fn resource_datum_builtin(
    path: &str,
    fields: &[&str],
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse(path).map_err(|error| error.to_string())?);
    for (field, value) in fields.iter().zip(arguments) {
        state
            .heap_mut()
            .set_datum_field(
                datum,
                FieldName::parse(field).map_err(|error| error.to_string())?,
                value.clone(),
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(Value::Datum(datum))
}

fn newlist_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let result = state.heap.allocate_list();
    for argument in arguments {
        let Value::TypePath(path) = argument else {
            return Err("newlist() arguments must be type paths".to_owned());
        };
        let datum = state.heap.allocate_datum(path.clone());
        state
            .heap
            .list_mut(result)
            .map_err(|error| error.to_string())?
            .add(Value::Datum(datum));
    }
    Ok(Value::List(result))
}

fn values_cut(
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
            .map_or(true, |value| {
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

fn values_dot(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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

fn values_fold(
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

/// Implements Dream Maker's legacy `text()` template form. Empty bracket
/// expressions in the literal template consume the following arguments in
/// order; whitespace inside a hole is ignored. Escaped brackets remain text.
fn text_template(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(Value::Text(template)) = arguments.first() else {
        return Err("text() expected a string as its first argument".to_owned());
    };
    let mut values = arguments[1..].iter();
    let mut output = String::with_capacity(template.len());
    let mut characters = template.chars().peekable();
    let mut holes = 0_usize;
    while let Some(character) = characters.next() {
        if character == '\\' && matches!(characters.peek(), Some('[' | ']')) {
            output.push(characters.next().expect("peeked escaped bracket exists"));
            continue;
        }
        if character != '[' {
            output.push(character);
            continue;
        }

        let mut lookahead = characters.clone();
        let mut whitespace = String::new();
        while lookahead.peek().is_some_and(|value| value.is_whitespace()) {
            whitespace.push(lookahead.next().expect("peeked whitespace exists"));
        }
        if lookahead.next() != Some(']') {
            output.push('[');
            continue;
        }
        for _ in 0..whitespace.chars().count() + 1 {
            characters.next();
        }
        let value = values
            .next()
            .ok_or_else(|| "text() has fewer arguments than template holes".to_owned())?;
        output.push_str(&runtime_text(value, state, "text() interpolation")?);
        holes += 1;
    }
    if values.next().is_some() {
        return Err(format!(
            "text() has more arguments than template holes ({holes})"
        ));
    }
    Ok(Value::text(output))
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

fn extrema_builtin(
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
        | Value::TypePath(_)
        | Value::ModifiedTypePath(_)
        | Value::Datum(_)
        | Value::List(_) => 0.0,
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
        Value::TypePath(_) | Value::ModifiedTypePath(_) | Value::Datum(_) | Value::List(_) => true,
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

fn del_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    match &arguments[0] {
        Value::Null => {}
        Value::Datum(datum) => {
            unregister_runtime_datum(state, *datum)?;
            state
                .heap_mut()
                .destroy_datum(*datum)
                .map_err(|error| format!("del failed: {error}"))?;
        }
        Value::List(list) => {
            state.associative_lists.remove(list);
            state
                .heap_mut()
                .destroy_list(*list)
                .map_err(|error| format!("del failed: {error}"))?;
        }
        value => return Err(format!("del cannot delete {value}")),
    }
    Ok(Value::Null)
}

fn qdel_value(value: &Value, state: &mut ExecutionState) -> Result<(), String> {
    match value {
        Value::Null => Ok(()),
        Value::Number(_) | Value::Text(_) | Value::TypePath(_) | Value::ModifiedTypePath(_) => {
            Ok(())
        }
        Value::Datum(datum) => {
            unregister_runtime_datum(state, *datum)?;
            state
                .heap_mut()
                .destroy_datum(*datum)
                .map_err(|error| error.to_string())
                .map(|_| ())
        }
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

fn unregister_runtime_datum(state: &mut ExecutionState, datum: DatumId) -> Result<(), String> {
    let loc = FieldName::parse("loc").expect("built-in loc field");
    let old_loc = state
        .heap
        .datum_field(datum, &loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(loc) => Some(*loc),
            _ => None,
        });
    synchronize_moved_atom_contents(state, datum, old_loc, None)?;

    let world = FieldName::parse("world").expect("built-in world global");
    let contents = FieldName::parse("contents").expect("built-in contents field");
    let world_contents = state
        .global(&world)
        .and_then(|value| match value {
            Value::Datum(world) => Some(*world),
            _ => None,
        })
        .and_then(|world| state.heap.datum_field(world, &contents).ok())
        .and_then(|value| match value {
            Value::List(list) => Some(*list),
            _ => None,
        });
    if let Some(list) = world_contents {
        state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .remove_first(&Value::Datum(datum));
    }
    Ok(())
}

fn typecacheof_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let target = arguments
        .first()
        .ok_or_else(|| "typecacheof requires a base type".to_owned())?;
    let raw_targets = match target {
        Value::List(list) => state
            .heap()
            .list(*list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        target => vec![target.clone()],
    };
    let targets = raw_targets
        .iter()
        .filter_map(|target| match target {
            // DM's typesof(null) contributes no paths. This matters for helper
            // lists which deliberately contain conditional/null entries.
            Value::Null => None,
            Value::TypePath(path) => Some(Ok(path.clone())),
            Value::ModifiedTypePath(path) => Some(Ok(path.base().clone())),
            Value::Text(text) => Some(
                TypePath::parse(text)
                    .map_err(|_| format!("typecacheof requires type paths, received {target}")),
            ),
            _ => Some(Err(format!(
                "typecacheof requires type paths, received {target}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let paths = {
        let mut paths = std::collections::BTreeSet::new();
        for target in targets {
            paths.insert(target.clone());
            paths.extend(
                state
                    .type_paths()
                    .filter(|path| {
                        *path == &target
                            || path.as_str().starts_with(&format!("{}/", target.as_str()))
                    })
                    .cloned(),
            );
        }
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

    for (name, value) in [
        ("alpha", Value::number(255.0)),
        ("appearance_flags", Value::number(0.0)),
        ("blend_mode", Value::number(0.0)),
        ("color", Value::Null),
        ("dir", Value::number(2.0)),
        ("icon", Value::Null),
        ("icon_state", Value::Null),
        ("layer", Value::number(0.0)),
        ("loc", Value::Null),
        ("name", Value::Null),
        ("overlays", Value::Null),
        ("plane", Value::number(0.0)),
        ("transform", Value::Null),
        ("underlays", Value::Null),
        ("vis_contents", Value::Null),
    ] {
        let _ = datum.set_field(FieldName::parse(name).expect("image field name"), value);
    }

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

fn regex_quote(
    arguments: &[Value],
    state: &ExecutionState,
    replacement: bool,
) -> Result<Value, String> {
    let text = arguments
        .first()
        .map(|value| runtime_text(value, state, "REGEX_QUOTE argument"))
        .transpose()?
        .unwrap_or_default();
    if replacement {
        return Ok(Value::text(text.replace('$', "$$")));
    }
    let mut quoted = String::with_capacity(text.len());
    for character in text.chars() {
        if matches!(
            character,
            '\\' | '.' | '^' | '$' | '|' | '?' | '*' | '+' | '(' | ')' | '[' | ']' | '{' | '}'
        ) {
            quoted.push('\\');
        }
        quoted.push(character);
    }
    Ok(Value::text(quoted))
}

fn headless_browse(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let descriptor = state.heap.allocate_list();
    let list = state
        .heap
        .list_mut(descriptor)
        .expect("new browse descriptor is live");
    list.set_key(
        Value::text("body"),
        arguments.first().cloned().unwrap_or(Value::Null),
    );
    list.set_key(
        Value::text("options"),
        arguments.get(1).cloned().unwrap_or(Value::Null),
    );
    Ok(Value::List(descriptor))
}

fn headless_winset(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let Some(Value::Datum(client)) = arguments.first() else {
        // BYOND accepts null when no client is available; a headless server
        // has no window to mutate in that case.
        return Ok(Value::Null);
    };
    let field = FieldName::parse("_dream64_winset").expect("headless UI field is valid");
    let settings = match state.heap.datum_field(*client, &field) {
        Ok(Value::List(settings)) => *settings,
        _ => {
            let settings = state.heap.allocate_list();
            state
                .heap
                .set_datum_field(*client, field, Value::List(settings))
                .map_err(|error| error.to_string())?;
            settings
        }
    };
    let control = arguments.get(1).cloned().unwrap_or(Value::Null);
    let params = arguments.get(2).cloned().unwrap_or(Value::Null);
    state
        .heap
        .list_mut(settings)
        .map_err(|error| error.to_string())?
        .set_key(control, params);
    Ok(Value::Null)
}

fn headless_winexists(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(Value::Datum(client)) = arguments.first() else {
        return Ok(Value::number(0.0));
    };
    let control = arguments.get(1).cloned().unwrap_or(Value::Null);
    let exists = state
        .heap
        .datum_field(
            *client,
            &FieldName::parse("_dream64_winset").expect("headless UI field is valid"),
        )
        .ok()
        .and_then(|value| match value {
            Value::List(settings) => state.heap.list(*settings).ok(),
            _ => None,
        })
        .is_some_and(|settings| settings.get_key(&control).is_ok());
    Ok(Value::number(if exists { 1.0 } else { 0.0 }))
}

/// A headless server cannot display BYOND's modal alert window. Select the
/// first offered button, which is the deterministic analogue of accepting the
/// dialog's default action. Both documented call forms are accepted:
/// `alert(usr, message, title, button1, ...)` and the implicit-usr form.
fn headless_alert(arguments: &[Value]) -> Result<Value, String> {
    let explicit_usr =
        arguments.len() >= 4 && matches!(arguments.first(), Some(Value::Datum(_) | Value::Null));
    let button = arguments
        .get(if explicit_usr { 3 } else { 2 })
        .filter(|value| !matches!(value, Value::Null))
        .cloned()
        .unwrap_or_else(|| Value::text("Ok"));
    Ok(button)
}

fn floor_multiple(arguments: &[Value]) -> Result<Value, String> {
    let value = arguments[0]
        .as_number()
        .ok_or_else(|| format!("FLOOR value must be numeric, received {}", arguments[0]))?;
    let multiple = arguments[1]
        .as_number()
        .ok_or_else(|| format!("FLOOR multiple must be numeric, received {}", arguments[1]))?;
    if multiple == 0.0 {
        return Ok(Value::number(0.0));
    }
    Ok(Value::number((value / multiple).floor() * multiple))
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

fn spatial_query(
    arguments: &[Value],
    state: &mut ExecutionState,
    mobs_only: bool,
    exclude_center: bool,
) -> Result<Value, String> {
    let (distance, center) = match arguments {
        [center] => (5.0, center),
        [distance, center] => (number(distance, "spatial query distance")?.floor(), center),
        _ => return Err("spatial query requires a center and optional distance".to_owned()),
    };
    let output = state.heap.allocate_list();
    let Some((center_x, center_y, center_z)) = datum_coordinates(state, center) else {
        return Ok(Value::List(output));
    };
    if !distance.is_finite() || distance < 0.0 {
        return Ok(Value::List(output));
    }
    let matching = state
        .heap
        .datums()
        .filter_map(|(id, datum)| {
            if exclude_center && matches!(center, Value::Datum(center_id) if *center_id == id) {
                return None;
            }
            let path = datum.type_path().as_str();
            if path == "/area" || path.starts_with("/area/") {
                return None;
            }
            if mobs_only && path != "/mob" && !path.starts_with("/mob/") {
                return None;
            }
            let coordinate = |name: &str| {
                datum
                    .field(&FieldName::parse(name).expect("coordinate field"))
                    .ok()?
                    .as_number()
            };
            let (x, y, z) = (coordinate("x")?, coordinate("y")?, coordinate("z")?);
            (z == center_z && (x - center_x).abs() <= distance && (y - center_y).abs() <= distance)
                .then_some(id)
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

fn step_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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
    let turf = state.heap.datums().find_map(|(id, datum)| {
        let path = datum.type_path().as_str();
        if path != "/turf" && !path.starts_with("/turf/") {
            return None;
        }
        let coordinate = |name: &str| {
            datum
                .field(&FieldName::parse(name).expect("coordinate field"))
                .ok()?
                .as_number()
        };
        ((coordinate("x")?, coordinate("y")?, coordinate("z")?) == target).then_some(id)
    });
    let Some(turf) = turf else {
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
    if old_loc != Some(turf) {
        synchronize_moved_atom_contents(state, atom, old_loc, Some(turf))?;
    }
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
    Ok(Value::number(1.0))
}

pub(super) fn synchronize_moved_atom_contents(
    state: &mut ExecutionState,
    atom: DatumId,
    old_loc: Option<DatumId>,
    new_loc: Option<DatumId>,
) -> Result<(), String> {
    let contents = FieldName::parse("contents").expect("built-in contents field");
    let loc = FieldName::parse("loc").expect("built-in loc field");
    let enclosing_area = |state: &ExecutionState, turf: DatumId| {
        state
            .heap
            .datum_field(turf, &loc)
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
            .datum_field(container, &contents)
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
    if let Some(list) = new_loc.and_then(|container| contents_list(state, container)) {
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
    if state.is_associative_list(list) {
        state.mark_associative_list(result);
    }
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
    let root = state
        .project_root()
        .ok_or_else(|| format!("{context} requires a configured project root"))?;
    let root = root
        .canonicalize()
        .map_err(|error| format!("{context} project root is unavailable: {error}"))?;
    let candidate = if path.is_absolute() {
        path
    } else {
        if path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return Err(format!("{context} path escapes the project root"));
        }
        root.join(path)
    };
    let existing = if candidate.exists() {
        candidate
            .canonicalize()
            .map_err(|error| error.to_string())?
    } else {
        let parent = candidate
            .parent()
            .ok_or_else(|| format!("{context} path has no parent"))?;
        let parent = parent
            .canonicalize()
            .map_err(|error| format!("{context} parent directory is unavailable: {error}"))?;
        parent.join(
            candidate
                .file_name()
                .ok_or_else(|| format!("{context} path is invalid"))?,
        )
    };
    if !existing.starts_with(&root) {
        return Err(format!("{context} path escapes the project root"));
    }
    Ok(existing)
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

fn fdel(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let raw = strict_text(&arguments[0], state, "fdel")?;
    let directory = raw.ends_with('/') || raw.ends_with('\\');
    let path = resolved_file_path(arguments, state, "fdel")?;
    let result = if directory {
        fs::remove_dir(path)
    } else {
        fs::remove_file(path)
    };
    Ok(Value::number(f32::from(result.is_ok())))
}

fn text2file(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text2file text")?;
    let path = resolved_file_path(&arguments[1..], state, "text2file")?;
    let append = arguments.get(2).is_some_and(truthy);
    let mut options = OpenOptions::new();
    options
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append);
    options
        .open(path)
        .and_then(|mut file| file.write_all(text.as_bytes()))
        .map_err(|error| format!("text2file failed: {error}"))?;
    Ok(Value::number(1.0))
}

fn fcopy(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let source = resolved_file_path(arguments, state, "fcopy source")?;
    let destination = resolved_file_path(&arguments[1..], state, "fcopy destination")?;
    Ok(Value::number(f32::from(
        fs::copy(source, destination).is_ok(),
    )))
}

fn flist(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let fallback = [Value::text(".")];
    let path = resolved_file_path(
        if arguments.is_empty() {
            &fallback
        } else {
            arguments
        },
        state,
        "flist",
    )?;
    let list = state.heap_mut().allocate_list();
    let mut names = fs::read_dir(path)
        .map_err(|error| format!("flist failed: {error}"))?
        .filter_map(Result::ok)
        .map(|entry| {
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                name.push('/');
            }
            name
        })
        .collect::<Vec<_>>();
    names.sort();
    for name in names {
        state
            .heap_mut()
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .add(Value::text(name));
    }
    Ok(Value::List(list))
}

pub(super) fn execute_output(
    target: &Value,
    value: &Value,
    state: &mut ExecutionState,
) -> Result<(), String> {
    let Value::Text(_) = target else {
        return Ok(());
    };
    let path = resolved_file_path(std::slice::from_ref(target), state, "output")?;
    let text = runtime_text(value, state, "output value")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("output failed: {error}"))?;
    writeln!(file, "{text}").map_err(|error| format!("output failed: {error}"))
}

fn html_encode(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "html_encode")?;
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        output.push_str(match character {
            '&' => "&amp;",
            '<' => "&lt;",
            '>' => "&gt;",
            '"' => "&quot;",
            '\'' => "&#39;",
            _ => {
                output.push(character);
                continue;
            }
        });
    }
    Ok(Value::text(output))
}

fn html_decode(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "html_decode")?;
    Ok(Value::text(
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&#x27;", "'")
            .replace("&amp;", "&"),
    ))
}

fn color_byte(value: &Value, context: &str) -> Result<u8, String> {
    Ok(number(value, context)?.round().clamp(0.0, 255.0) as u8)
}

fn rgb_builtin(arguments: &[Value]) -> Result<Value, String> {
    let r = color_byte(&arguments[0], "rgb red")?;
    let g = color_byte(&arguments[1], "rgb green")?;
    let b = color_byte(&arguments[2], "rgb blue")?;
    // The fifth positional argument is color space. RGB is the native/default
    // space; conversion of alternate spaces is kept explicit rather than
    // silently producing the wrong color.
    if arguments.len() == 5 && arguments[4].as_number().is_some_and(|space| space != 0.0) {
        return Err("rgb alternate color spaces are not implemented".to_owned());
    }
    if let Some(alpha) = arguments.get(3) {
        Ok(Value::text(format!(
            "#{r:02x}{g:02x}{b:02x}{:02x}",
            color_byte(alpha, "rgb alpha")?
        )))
    } else {
        Ok(Value::text(format!("#{r:02x}{g:02x}{b:02x}")))
    }
}

fn parse_hex_color(text: &str) -> Option<Vec<u8>> {
    let hex = text.strip_prefix('#')?;
    let expanded = match hex.len() {
        3 | 4 => hex.chars().flat_map(|c| [c, c]).collect::<String>(),
        6 | 8 => hex.to_owned(),
        _ => return None,
    };
    (0..expanded.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&expanded[index..index + 2], 16).ok())
        .collect()
}

fn rgb2num_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments
        .get(1)
        .and_then(Value::as_number)
        .is_some_and(|space| space != 0.0)
    {
        return Err("rgb2num alternate color spaces are not implemented".to_owned());
    }
    let text = strict_text(&arguments[0], state, "rgb2num")?;
    let components =
        parse_hex_color(&text).ok_or_else(|| format!("rgb2num invalid color {text:?}"))?;
    let id = state.heap.allocate_list();
    let list = state.heap.list_mut(id).map_err(|error| error.to_string())?;
    for component in components {
        list.add(Value::number(f32::from(component)));
    }
    Ok(Value::List(id))
}

fn gradient_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let mut index = number(arguments.last().expect("gradient arity"), "gradient index")?;
    let items = &arguments[..arguments.len() - 1];
    let mut stops = Vec::new();
    let mut looping = false;
    if items
        .first()
        .is_some_and(|value| value.as_number().is_some())
    {
        let mut cursor = 0;
        while cursor + 1 < items.len() {
            let Some(position) = items[cursor].as_number() else {
                break;
            };
            if !matches!(items[cursor + 1], Value::Text(_)) {
                break;
            }
            stops.push((position, &items[cursor + 1]));
            cursor += 2;
        }
        looping = items[cursor..]
            .iter()
            .any(|value| matches!(value, Value::Text(text) if text.eq_ignore_ascii_case("loop")));
    } else {
        let colors = items
            .iter()
            .filter(|value| matches!(value, Value::Text(_)))
            .collect::<Vec<_>>();
        let divisor = colors.len().saturating_sub(1).max(1) as f32;
        stops.extend(
            colors
                .into_iter()
                .enumerate()
                .map(|(i, color)| (i as f32 / divisor, color)),
        );
    }
    if stops.len() < 2 {
        return Err("gradient requires at least two color stops".to_owned());
    }
    let first = stops[0].0;
    let last = stops[stops.len() - 1].0;
    if looping && last > first {
        index = (index - first).rem_euclid(last - first) + first;
    }
    let segment = stops
        .windows(2)
        .position(|pair| index <= pair[1].0)
        .unwrap_or(stops.len() - 2);
    let (left_at, left_value) = stops[segment];
    let (right_at, right_value) = stops[segment + 1];
    let amount = if right_at == left_at {
        0.0
    } else {
        (index - left_at) / (right_at - left_at)
    };
    let left = parse_hex_color(&strict_text(left_value, state, "gradient color")?)
        .ok_or_else(|| "gradient requires hexadecimal colors".to_owned())?;
    let right = parse_hex_color(&strict_text(right_value, state, "gradient color")?)
        .ok_or_else(|| "gradient requires hexadecimal colors".to_owned())?;
    let count = left.len().max(right.len());
    let mut output = String::from("#");
    for component in 0..count {
        let a = f32::from(*left.get(component).unwrap_or(&255));
        let b = f32::from(*right.get(component).unwrap_or(&255));
        write!(output, "{:02x}", (a + (b - a) * amount).round() as u8).unwrap();
    }
    Ok(Value::text(output))
}

fn time2text_builtin(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let ticks = number(&arguments[0], "time2text timestamp")? as i64;
    let format = arguments.get(1).map_or_else(
        || Ok("DDD MMM DD hh:mm:ss YYYY".to_owned()),
        |value| strict_text(value, state, "time2text format"),
    )?;
    let timezone = arguments.get(2).and_then(Value::as_number).unwrap_or(0.0);
    let seconds = ticks.div_euclid(10) + (timezone * 3600.0) as i64;
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days_since_2000(days);
    let weekdays = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"];
    let weekday_names = [
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let hour = day_seconds / 3600;
    let minute = day_seconds / 60 % 60;
    let second = day_seconds % 60;
    let mut out = format;
    for (token, value) in [
        ("YYYY", format!("{year:04}")),
        ("Month", month_names[month - 1].to_owned()),
        ("DDD", weekdays[days.rem_euclid(7) as usize].to_owned()),
        ("Day", weekday_names[days.rem_euclid(7) as usize].to_owned()),
        ("MMM", months[month - 1].to_owned()),
        ("YY", format!("{:02}", year % 100)),
        ("MM", format!("{month:02}")),
        ("DD", format!("{day:02}")),
        ("hh", format!("{hour:02}")),
        ("mm", format!("{minute:02}")),
        ("ss", format!("{second:02}")),
    ] {
        out = out.replace(token, &value);
    }
    Ok(Value::text(out))
}

fn civil_from_days_since_2000(days: i64) -> (i64, usize, i64) {
    // Howard Hinnant's civil date algorithm, offset from 1970 to 2000.
    let z = days + 10_957 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month as usize, day)
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

#[cfg(test)]
mod color_text_file_tests {
    use super::*;

    #[test]
    fn rgb_round_trips_short_and_alpha_hex_colors() {
        let mut state = ExecutionState::new();
        assert_eq!(
            rgb_builtin(&[Value::number(255.0), Value::number(128.0), Value::Null]).unwrap(),
            Value::text("#ff8000")
        );
        let Value::List(parts) = rgb2num_builtin(&[Value::text("#5af8")], &mut state).unwrap()
        else {
            panic!("rgb2num must return a list")
        };
        let parts = state.heap.list(parts).unwrap();
        assert_eq!(parts.get(1), Ok(&Value::number(85.0)));
        assert_eq!(parts.get(2), Ok(&Value::number(170.0)));
        assert_eq!(parts.get(3), Ok(&Value::number(255.0)));
        assert_eq!(parts.get(4), Ok(&Value::number(136.0)));
    }

    #[test]
    fn gradient_interpolates_rgb_components() {
        let mut state = ExecutionState::new();
        assert_eq!(
            gradient_builtin(
                &[
                    Value::text("#ff0000"),
                    Value::text("#000000"),
                    Value::number(0.2)
                ],
                &mut state
            )
            .unwrap(),
            Value::text("#cc0000")
        );
        assert_eq!(
            gradient_builtin(
                &[
                    Value::number(0.0),
                    Value::text("#ff0000"),
                    Value::number(1.0),
                    Value::text("#000000"),
                    Value::text("loop"),
                    Value::number(0.2),
                ],
                &mut state,
            )
            .unwrap(),
            Value::text("#cc0000")
        );
    }

    #[test]
    fn html_entities_round_trip_without_double_decoding() {
        let state = ExecutionState::new();
        let encoded = html_encode(&[Value::text("<&\"'>")], &state).unwrap();
        assert_eq!(encoded, Value::text("&lt;&amp;&quot;&#39;&gt;"));
        assert_eq!(
            html_decode(&[encoded], &state).unwrap(),
            Value::text("<&\"'>")
        );
    }

    #[test]
    fn realtime_epoch_and_timezone_format_deterministically() {
        let state = ExecutionState::new();
        assert_eq!(
            time2text_builtin(
                &[
                    Value::number(0.0),
                    Value::text("YYYY-MM-DD hh:mm:ss"),
                    Value::number(0.0)
                ],
                &state
            )
            .unwrap(),
            Value::text("2000-01-01 00:00:00")
        );
        assert_eq!(
            time2text_builtin(
                &[
                    Value::number(0.0),
                    Value::text("hh:mm"),
                    Value::number(-5.0)
                ],
                &state
            )
            .unwrap(),
            Value::text("19:00")
        );
    }

    #[test]
    fn filesystem_builtins_and_output_stay_inside_project_root() {
        let root = std::env::temp_dir().join(format!("dream64-vm-files-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("data/logs")).unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());

        assert_eq!(
            text2file(
                &[Value::text("first"), Value::text("data/logs/runtime.log")],
                &state
            )
            .unwrap(),
            Value::number(1.0)
        );
        execute_output(
            &Value::text("data/logs/runtime.log"),
            &Value::text("second"),
            &mut state,
        )
        .unwrap();
        assert_eq!(
            file2text(&[Value::text("data/logs/runtime.log")], &state).unwrap(),
            Value::text("firstsecond\n")
        );
        assert_eq!(
            fcopy(
                &[
                    Value::text("data/logs/runtime.log"),
                    Value::text("data/logs/copy.log")
                ],
                &state
            )
            .unwrap(),
            Value::number(1.0)
        );
        let Value::List(files) = flist(&[Value::text("data/logs")], &mut state).unwrap() else {
            panic!("flist should return a list");
        };
        assert_eq!(state.heap().list(files).unwrap().len(), 2);
        assert!(fexists(&[Value::text("../outside")], &state).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rust_g_file_bridge_overwrites_appends_creates_and_rejects_traversal() {
        let root = std::env::temp_dir().join(format!(
            "dream64-rust-g-files-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("data")).unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let library = Value::text("rust_g");

        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("file_write"),
                &[Value::text("first"), Value::text("data/runtime.log")],
                &mut state,
            ),
            Ok(Value::Null)
        );
        assert_eq!(
            fs::read_to_string(root.join("data/runtime.log")).unwrap(),
            "first"
        );
        execute_external_call(
            &library,
            &Value::text("file_append"),
            &[Value::text("+second"), Value::text("data/runtime.log")],
            &mut state,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(root.join("data/runtime.log")).unwrap(),
            "first+second"
        );
        execute_external_call(
            &library,
            &Value::text("file_write"),
            &[Value::text("replacement"), Value::text("data/runtime.log")],
            &mut state,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(root.join("data/runtime.log")).unwrap(),
            "replacement"
        );
        assert!(
            execute_external_call(
                &library,
                &Value::text("file_write"),
                &[Value::text("escape"), Value::text("../escape.log")],
                &mut state,
            )
            .is_err()
        );
        let outside = std::env::temp_dir().join(format!(
            "dream64-rust-g-outside-{}-{:?}.log",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::write(&outside, "outside").unwrap();
        let link = root.join("data/linked.log");
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_file(&outside, &link);
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&outside, &link);
        if linked.is_ok() {
            assert!(
                execute_external_call(
                    &library,
                    &Value::text("file_write"),
                    &[Value::text("escape"), Value::text("data/linked.log")],
                    &mut state,
                )
                .is_err(),
                "an existing symlink may not redirect writes outside the project root"
            );
            assert_eq!(fs::read_to_string(&outside).unwrap(), "outside");
        }
        assert!(execute_external_call(&library, &Value::text("unknown"), &[], &mut state).is_err());
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(outside).unwrap();
    }

    #[test]
    fn rust_g_toml_bridge_returns_double_encoded_config_envelope() {
        let root = std::env::temp_dir().join(format!(
            "dream64-rust-g-toml-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("config")).unwrap();
        fs::write(root.join("config/settings.toml"), "# config\n[shared]\n\"# phrase\" = \"blocked # text\"\nenabled = true\nweights = [1, -2, 3.5]\n[server.network]\nport = 1337\n[[relay]]\nid = \"east\"\naddress = \"byond://east:{port}\"\n[[relay]]\nid = \"direct\"\naddress = \"byond://direct:{port}\"\n").unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let Value::Text(envelope) = execute_external_call(
            &Value::text("rust_g"),
            &Value::text("toml_file_to_json"),
            &[Value::text("config/settings.toml")],
            &mut state,
        )
        .unwrap() else {
            panic!("TOML bridge should return text")
        };
        let envelope: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        assert_eq!(envelope["success"], true);
        let document: serde_json::Value =
            serde_json::from_str(envelope["content"].as_str().unwrap()).unwrap();
        assert_eq!(document["shared"]["# phrase"], "blocked # text");
        assert_eq!(document["shared"]["weights"][2], 3.5);
        assert_eq!(document["server"]["network"]["port"], 1337);
        assert_eq!(document["relay"][1]["id"], "direct");

        let Value::Text(missing) = execute_external_call(
            &Value::text("rust_g"),
            &Value::text("toml_file_to_json"),
            &[Value::text("config/missing.toml")],
            &mut state,
        )
        .unwrap() else {
            unreachable!()
        };
        let missing: serde_json::Value = serde_json::from_str(&missing).unwrap();
        assert_eq!(missing["success"], false);
        assert!(!missing["content"].as_str().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rust_g_named_timers_reset_and_return_numeric_text() {
        let mut state = ExecutionState::new();
        let library = Value::text("rust_g");
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("time_reset"),
                &[Value::text("subsystem")],
                &mut state,
            ),
            Ok(Value::Null)
        );
        let Value::Text(milliseconds) = execute_external_call(
            &library,
            &Value::text("time_milliseconds"),
            &[Value::text("subsystem")],
            &mut state,
        )
        .unwrap() else {
            panic!("timer should return numeric text")
        };
        assert!(milliseconds.parse::<f64>().is_ok());
    }
}

#[cfg(test)]
mod spatial_tests {
    use super::*;

    fn place(state: &mut ExecutionState, path: &str, x: f32, y: f32) -> dm_value::DatumId {
        let id = state.heap.allocate_datum(TypePath::parse(path).unwrap());
        for (name, value) in [("x", x), ("y", y), ("z", 1.0)] {
            state
                .heap
                .set_datum_field(id, FieldName::parse(name).unwrap(), Value::number(value))
                .unwrap();
        }
        id
    }

    #[test]
    fn view_families_filter_distance_center_and_mob_type() {
        let mut state = ExecutionState::new();
        let center = place(&mut state, "/turf/open", 5.0, 5.0);
        place(&mut state, "/mob/living", 6.0, 5.0);
        place(&mut state, "/obj/item", 6.0, 6.0);
        place(&mut state, "/mob/living", 9.0, 5.0);
        let Value::List(view) = spatial_query(
            &[Value::number(1.0), Value::Datum(center)],
            &mut state,
            false,
            false,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(state.heap.list(view).unwrap().len(), 3);
        let Value::List(viewers) = spatial_query(
            &[Value::number(1.0), Value::Datum(center)],
            &mut state,
            true,
            false,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(state.heap.list(viewers).unwrap().len(), 1);
        let Value::List(oview) = spatial_query(
            &[Value::number(1.0), Value::Datum(center)],
            &mut state,
            false,
            true,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(state.heap.list(oview).unwrap().len(), 2);
    }

    #[test]
    fn step_moves_to_a_materialized_neighbor_and_reports_failure() {
        let mut state = ExecutionState::new();
        let origin = place(&mut state, "/turf/open", 2.0, 2.0);
        let east = place(&mut state, "/turf/open", 3.0, 2.0);
        let mob = place(&mut state, "/mob/living", 2.0, 2.0);
        let west_area = place(&mut state, "/area/west", 0.0, 0.0);
        let east_area = place(&mut state, "/area/east", 0.0, 0.0);
        let contents = FieldName::parse("contents").unwrap();
        for datum in [origin, east, west_area, east_area] {
            let list = state.heap.allocate_list();
            state
                .heap
                .set_datum_field(datum, contents.clone(), Value::List(list))
                .unwrap();
        }
        state
            .heap
            .list_mut(match state.heap.datum_field(origin, &contents).unwrap() {
                Value::List(list) => *list,
                _ => unreachable!(),
            })
            .unwrap()
            .add(Value::Datum(mob));
        for (datum, loc) in [(origin, west_area), (east, east_area), (mob, origin)] {
            state
                .heap
                .set_datum_field(datum, FieldName::parse("loc").unwrap(), Value::Datum(loc))
                .unwrap();
        }
        let west_contents = match state.heap.datum_field(west_area, &contents).unwrap() {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        let east_contents = match state.heap.datum_field(east_area, &contents).unwrap() {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        state
            .heap
            .list_mut(west_contents)
            .unwrap()
            .add(Value::Datum(mob));
        assert_eq!(
            step_builtin(&[Value::Datum(mob), Value::number(4.0)], &mut state).unwrap(),
            Value::number(1.0)
        );
        assert_eq!(
            state
                .heap
                .datum(mob)
                .unwrap()
                .field(&FieldName::parse("loc").unwrap()),
            Ok(&Value::Datum(east))
        );
        let origin_contents = match state.heap.datum_field(origin, &contents).unwrap() {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        let east_turf_contents = match state.heap.datum_field(east, &contents).unwrap() {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        assert!(
            !state
                .heap
                .list(origin_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert!(
            state
                .heap
                .list(east_turf_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert!(
            !state
                .heap
                .list(west_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert!(
            state
                .heap
                .list(east_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert_eq!(
            step_builtin(&[Value::Datum(mob), Value::number(4.0)], &mut state).unwrap(),
            Value::number(0.0)
        );
        del_builtin(&[Value::Datum(mob)], &mut state).unwrap();
        assert!(
            !state
                .heap
                .list(east_turf_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert!(
            !state
                .heap
                .list(east_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert_ne!(origin, east);
    }
}
