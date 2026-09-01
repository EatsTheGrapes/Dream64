//! Timestamp stamping, `iconforge`/`GAGS`, git bridge, and the embedded TOML
//! reader used by `world` config and icon-forge `.toml` layers.

use std::fs;
use std::path::Component;
use std::process::Command;

use dm_value::Value;

use super::{ExecutionState, resolved_file_path, strict_text};
pub(super) fn format_unix_timestamp(unix_millis: i64, format: &str, offset_hours: f32) -> String {
    let offset_seconds = (offset_hours * 3_600.0).round() as i64;
    let local_millis = unix_millis.saturating_add(offset_seconds.saturating_mul(1_000));
    let days = local_millis.div_euclid(86_400_000);
    let day_millis = local_millis.rem_euclid(86_400_000);
    let (year, month, day) = civil_date_from_unix_days(days);
    let hour = day_millis / 3_600_000;
    let minute = day_millis / 60_000 % 60;
    let second = day_millis / 1_000 % 60;
    let millis = day_millis % 1_000;
    let offset_sign = if offset_seconds < 0 { '-' } else { '+' };
    let offset_abs = offset_seconds.abs();
    let offset = format!(
        "{offset_sign}{:02}{:02}",
        offset_abs / 3_600,
        offset_abs / 60 % 60
    );
    let literal_percent = "\u{0}";
    format
        .replace("%%", literal_percent)
        .replace("%.3f", &format!(".{millis:03}"))
        .replace("%F", &format!("{year:04}-{month:02}-{day:02}"))
        .replace("%T", &format!("{hour:02}:{minute:02}:{second:02}"))
        .replace("%Y", &format!("{year:04}"))
        .replace("%m", &format!("{month:02}"))
        .replace("%d", &format!("{day:02}"))
        .replace("%H", &format!("{hour:02}"))
        .replace("%M", &format!("{minute:02}"))
        .replace("%S", &format!("{second:02}"))
        .replace("%z", &offset)
        .replace(literal_percent, "%")
}

fn civil_date_from_unix_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

pub(super) fn owned_value_text(value: Value) -> String {
    match value {
        Value::Text(text) => text.to_string(),
        Value::Null => String::new(),
        value => value.to_string(),
    }
}

pub(super) fn iconforge_load_gags_config(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let config_path = strict_text(&arguments[0], state, "iconforge config path")?;
    let config_json = strict_text(&arguments[1], state, "iconforge config JSON")?;
    if let Err(error) = serde_json::from_str::<serde_json::Value>(&config_json) {
        return Ok(Value::text(format!(
            "IconForge error: Failed to parse config for '{config_path}': {error}"
        )));
    }
    let icon_path_text = strict_text(&arguments[2], state, "iconforge icon path")?;
    let icon_path = resolved_file_path(&arguments[2..3], state, "iconforge icon path")?;
    if let Err(error) = fs::metadata(&icon_path) {
        return Ok(Value::text(format!(
            "IconForge error: Failed to open DMI '{icon_path_text}' (resolved to '{}') - {error}",
            icon_path.display()
        )));
    }
    state.load_iconforge_gags_config(config_path, icon_path, config_json);
    Ok(Value::text("OK"))
}

pub(super) fn iconforge_gags(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let config_path = strict_text(&arguments[0], state, "iconforge config path")?;
    if !state.has_iconforge_gags_config(&config_path) {
        return Ok(Value::text(format!(
            "IconForge error: Provided config_path {config_path} has not been loaded by iconforge_load_gags_config!"
        )));
    }
    let output_text = strict_text(&arguments[2], state, "iconforge output path")?;
    let relative = std::path::Path::new(&output_text);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return Err("iconforge output path escapes the project root".to_owned());
    }
    let root = state
        .project_root()
        .ok_or_else(|| "iconforge output path requires a configured project root".to_owned())?
        .canonicalize()
        .map_err(|error| format!("iconforge project root is unavailable: {error}"))?;
    let mut parent = root.clone();
    for component in relative
        .parent()
        .into_iter()
        .flat_map(std::path::Path::components)
    {
        let Component::Normal(component) = component else {
            continue;
        };
        parent.push(component);
        if !parent.exists() {
            fs::create_dir(&parent).map_err(|error| {
                format!("IconForge error: Failed to create output directory: {error}")
            })?;
        }
        let resolved = parent.canonicalize().map_err(|error| {
            format!("IconForge error: Failed to resolve output directory: {error}")
        })?;
        if !resolved.starts_with(&root) {
            return Err("iconforge output path escapes the project root".to_owned());
        }
        parent = resolved;
    }
    let output = resolved_file_path(&arguments[2..3], state, "iconforge output path")?;
    // SS13 always joins the palette into a "#rrggbb#rrggbb" string before the
    // native call; tolerate anything else by falling back to an empty palette
    // so the output DMI still carries the config's state set.
    let colors = strict_text(&arguments[1], state, "iconforge colors").unwrap_or_default();
    let source = state
        .iconforge_gags_source(&config_path)
        .ok_or_else(|| format!("IconForge error: Config {config_path} lost its source DMI"))?
        .to_path_buf();
    let config_json = state
        .iconforge_gags_json(&config_path)
        .ok_or_else(|| format!("IconForge error: Config {config_path} lost its JSON"))?
        .to_owned();

    match composite_gags_dmi(&source, &config_json, &colors) {
        Ok(dmi_bytes) => {
            fs::write(&output, dmi_bytes).map_err(|error| {
                format!(
                    "IconForge error: Failed to write headless output '{}': {error}",
                    output.display()
                )
            })?;
            Ok(Value::text("OK"))
        }
        Err(error) => Ok(Value::text(format!(
            "IconForge error: GAGS compositing failed for {config_path}: {error}"
        ))),
    }
}

/// Composite a GAGS bundle DMI: load the template, render every output state's
/// layer stack with the supplied palette, and serialise the result.
fn composite_gags_dmi(
    template_path: &std::path::Path,
    config_json: &str,
    colors: &str,
) -> Result<Vec<u8>, String> {
    let template_bytes =
        fs::read(template_path).map_err(|error| format!("cannot read template DMI: {error}"))?;
    let template = dm_icon::IconBitmap::from_dmi_bytes(&template_bytes)
        .map_err(|error| format!("cannot decode template DMI: {error}"))?;
    let palette = dm_icon::gags::parse_color_string(colors);
    let output = dm_icon::gags::composite(config_json, &template, &palette)?;
    output
        .to_dmi_bytes()
        .map_err(|error| format!("cannot encode output DMI: {error}"))
}

pub(super) fn validate_git_revision(revision: &str) -> Result<(), String> {
    if revision.is_empty()
        || revision.len() > 256
        || revision.starts_with('-')
        || revision.contains("..")
        || revision.contains("//")
        || revision.chars().any(|character| {
            !(character.is_ascii_alphanumeric()
                || matches!(character, '/' | '.' | '_' | '-' | '^' | '~'))
        })
    {
        return Err("git revision contains unsafe syntax".to_owned());
    }
    Ok(())
}

pub(super) fn validate_git_date_format(format: &str) -> Result<(), String> {
    if format.is_empty()
        || format.len() > 128
        || format
            .chars()
            .any(|character| character.is_control() || matches!(character, '\0' | '\r' | '\n'))
    {
        return Err("git date format contains unsafe syntax".to_owned());
    }
    Ok(())
}

pub(super) fn run_git_bridge(state: &ExecutionState, arguments: &[&str]) -> Result<Value, String> {
    let root = state
        .project_root()
        .ok_or_else(|| "git bridge requires a configured project root".to_owned())?
        .canonicalize()
        .map_err(|error| format!("git bridge project root is unavailable: {error}"))?;
    let output = Command::new("git")
        .args(arguments)
        .current_dir(root)
        .output()
        .map_err(|error| format!("git bridge failed to start: {error}"))?;
    if !output.status.success() {
        return Ok(Value::Null);
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|error| format!("git bridge returned non-UTF-8 output: {error}"))?;
    Ok(Value::text(text.trim_end_matches(['\r', '\n']).to_owned()))
}

pub(super) fn parse_toml_document(source: &str) -> Result<serde_json::Value, String> {
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
