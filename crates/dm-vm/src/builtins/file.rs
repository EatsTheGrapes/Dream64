//! Filesystem procedures (`fexists`/`flist`/`fdel`/`file2text`/`fcopy`
//! and friends) plus `output`/`html_encode` streaming text.

use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, PathBuf};

use dm_value::{FieldName, Value};

use crate::LocalClientUiEvent;

use super::{
    ExecutionState, icons::icon_backing_resource, local_client_for_value, runtime_text,
    strict_text, truthy, value_text,
};
pub(crate) fn resolved_file_path(
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

pub(crate) fn relaxed_resolved_file_path(
    arguments: &[Value],
    state: &ExecutionState,
    context: &str,
) -> Result<PathBuf, String> {
    let raw = strict_text(&arguments[0], state, context)?;
    let relative = PathBuf::from(raw);
    let root = state
        .project_root()
        .ok_or_else(|| format!("{context} requires a configured project root"))?
        .canonicalize()
        .map_err(|error| format!("{context} project root is unavailable: {error}"))?;
    let candidate = if relative.is_absolute() {
        relative
    } else {
        if relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return Err(format!("{context} path escapes the project root"));
        }
        root.join(relative)
    };
    let mut ancestor = candidate.as_path();
    let resolved_ancestor = loop {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => break ancestor.canonicalize().map_err(|error| error.to_string())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| format!("{context} path has no existing ancestor"))?;
            }
            Err(error) => return Err(format!("{context} path is unavailable: {error}")),
        }
    };
    if !resolved_ancestor.starts_with(&root) {
        return Err(format!("{context} path escapes the project root"));
    }
    Ok(candidate)
}

pub(super) fn prepare_write_file_path(
    arguments: &[Value],
    state: &ExecutionState,
    context: &str,
) -> Result<Option<PathBuf>, String> {
    let candidate = relaxed_resolved_file_path(arguments, state, context)?;
    let parent = candidate
        .parent()
        .ok_or_else(|| format!("{context} path has no parent"))?;
    // BYOND creates every missing destination directory for its file-writing
    // builtins. Keep I/O failures as an ordinary false result for callers
    // such as fcopy()/text2file(), while retaining containment failures as
    // runtime errors rather than allowing a symlink escape.
    if fs::create_dir_all(parent).is_err() {
        return Ok(None);
    }
    let root = state
        .project_root()
        .ok_or_else(|| format!("{context} requires a configured project root"))?
        .canonicalize()
        .map_err(|error| format!("{context} project root is unavailable: {error}"))?;
    let parent = match parent.canonicalize() {
        Ok(parent) => parent,
        Err(_) => return Ok(None),
    };
    if !parent.starts_with(root) {
        return Err(format!("{context} path escapes the project root"));
    }
    Ok(Some(
        parent.join(
            candidate
                .file_name()
                .ok_or_else(|| format!("{context} path is invalid"))?,
        ),
    ))
}

pub(super) fn fexists(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let raw = strict_text(&arguments[0], state, "fexists")?;
    let relative = PathBuf::from(raw);
    let root = state
        .project_root()
        .ok_or_else(|| "fexists requires a configured project root".to_owned())?
        .canonicalize()
        .map_err(|error| format!("fexists project root is unavailable: {error}"))?;
    let invalid_relative_root = !relative.is_absolute()
        && relative
            .components()
            .any(|component| matches!(component, Component::RootDir | Component::Prefix(_)));
    if invalid_relative_root {
        return Err("fexists path escapes the project root".to_owned());
    }
    let path = if relative.is_absolute() {
        relative
    } else {
        let mut contained = PathBuf::new();
        for component in relative.components() {
            match component {
                Component::CurDir => {}
                Component::Normal(segment) => contained.push(segment),
                Component::ParentDir => {
                    if !contained.pop() {
                        return Err("fexists path escapes the project root".to_owned());
                    }
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err("fexists path escapes the project root".to_owned());
                }
            }
        }
        root.join(contained)
    };

    // A missing intermediate directory is an ordinary negative existence
    // result in BYOND. Canonicalize the nearest existing ancestor so that the
    // relaxed lookup still rejects symlink and absolute-path escapes.
    let mut ancestor = path.as_path();
    let resolved_ancestor = loop {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => break ancestor.canonicalize().map_err(|error| error.to_string())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| "fexists path has no existing ancestor".to_owned())?;
            }
            Err(error) => return Err(format!("fexists path is unavailable: {error}")),
        }
    };
    if !resolved_ancestor.starts_with(&root) {
        return Err("fexists path escapes the project root".to_owned());
    }
    Ok(Value::number(f32::from(path.exists())))
}

pub(super) fn file2text(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    // A contained path may have multiple nonexistent parent components. BYOND
    // reports a missing file as null; resolve its nearest existing ancestor
    // only to enforce root/symlink containment, then let the read return
    // NotFound normally.
    let path = relaxed_resolved_file_path(arguments, state, "file2text")?;
    // BYOND resources may name a directory (notably entries returned by
    // `flist()`). A directory is not readable file content, so `file2text()`
    // returns null instead of surfacing the host OS' access-denied/is-directory
    // error. OpenDream follows the same contract by only loading resource data
    // when `File.Exists(path)` is true.
    if !path.is_file() {
        return Ok(Value::Null);
    }
    match fs::read_to_string(path) {
        Ok(text) => Ok(Value::text(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(error) => Err(format!("file2text failed: {error}")),
    }
}

pub(super) fn fdel(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let raw = strict_text(&arguments[0], state, "fdel")?;
    let directory = raw.ends_with('/') || raw.ends_with('\\');
    let path = resolved_file_path(arguments, state, "fdel")?;
    let result = if directory {
        // BYOND treats a trailing slash as explicit authorization to remove
        // the entire directory tree, including nested files/directories.
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    Ok(Value::number(f32::from(result.is_ok())))
}

pub(super) fn text2file(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text2file text")?;
    let Some(path) = prepare_write_file_path(&arguments[1..], state, "text2file")? else {
        return Ok(Value::number(0.0));
    };
    // BYOND appends by default. A false optional compatibility flag requests
    // replacement, matching the existing extended arity accepted here.
    let append = arguments.get(2).is_none_or(truthy);
    let mut options = OpenOptions::new();
    options
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append);
    let result = options
        .open(path)
        .and_then(|mut file| file.write_all(text.as_bytes()));
    Ok(Value::number(f32::from(result.is_ok())))
}

pub(super) fn fcopy(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(mut source) = arguments.first().cloned() else {
        return Ok(Value::number(0.0));
    };
    if matches!(source, Value::Null) {
        return Ok(Value::number(0.0));
    }
    // A mutated /icon: composite its recorded raster journal into a real DMI
    // rather than copying the untouched backing resource.
    if let Value::Datum(icon) = source
        && super::icons::icon_journal_len(&state.heap, icon) > 0
        && let Ok(Some(bitmap)) = super::icons::materialize_icon_bitmap(icon, state, 0)
        && let Ok(dmi_bytes) = bitmap.to_dmi_bytes()
        && let Some(destination) =
            prepare_write_file_path(&arguments[1..], state, "fcopy destination")?
    {
        return Ok(Value::number(f32::from(
            fs::write(destination, dmi_bytes).is_ok(),
        )));
    }
    if let Value::Datum(_) = source {
        source = icon_backing_resource(&source, state, 0)?;
    }
    let source = match source {
        Value::Text(_) | Value::File(_) => {
            relaxed_resolved_file_path(&[source], state, "fcopy source")?
        }
        Value::Null => return Ok(Value::number(0.0)),
        value => {
            return Err(format!(
                "fcopy source requires text, received {}",
                runtime_text(&value, state, "fcopy source")?
            ));
        }
    };
    let Some(destination) = prepare_write_file_path(&arguments[1..], state, "fcopy destination")?
    else {
        return Ok(Value::number(0.0));
    };
    Ok(Value::number(f32::from(
        fs::copy(source, destination).is_ok(),
    )))
}

pub(super) fn flist(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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

pub(crate) fn execute_output(
    target: &Value,
    value: &Value,
    state: &mut ExecutionState,
) -> Result<(), String> {
    if let Value::Datum(target) = target {
        let routed_client = local_client_for_value(state, &Value::Datum(*target));
        if let Some(target) = routed_client
            && let Value::Datum(descriptor) = value
            && state.heap.datum(*descriptor).is_ok_and(|datum| {
                let path = datum.type_path().as_str();
                path == "/sound" || path.starts_with("/sound/")
            })
        {
            let field = |name: &str| {
                super::datum_field_or_initial(
                    state,
                    *descriptor,
                    &FieldName::parse(name).expect("sound field is valid"),
                )
                .map_err(|error| error.to_string())
            };
            let file = match field("file")? {
                Value::Null => None,
                Value::File(path) | Value::Text(path) => Some(path.to_string()),
                other => {
                    return Err(format!(
                        "sound file requires a resource path or null, received {other}"
                    ));
                }
            };
            let numeric = |name: &str, default: f32| -> Result<f32, String> {
                let value = field(name)?;
                Ok(value.as_number().unwrap_or(default))
            };
            state.emit_local_client_ui_event(
                target,
                LocalClientUiEvent::Sound {
                    file,
                    channel: numeric("channel", 0.0)? as i32,
                    repeat: truthy(&field("repeat")?),
                    volume: numeric("volume", 100.0)?.clamp(0.0, 100.0),
                    frequency: numeric("frequency", 0.0)?,
                    pan: numeric("pan", 0.0)?.clamp(-100.0, 100.0),
                },
            );
            return Ok(());
        }
        if let Some(target) = routed_client
            && let Value::Datum(descriptor) = value
            && state.heap.datum(*descriptor).is_ok_and(|datum| {
                let path = datum.type_path().as_str();
                path == "/output" || path.starts_with("/output/")
            })
        {
            let message = state
                .heap
                .datum_field(*descriptor, &FieldName::parse("message").unwrap())
                .map_err(|error| error.to_string())?
                .clone();
            let control = state
                .heap
                .datum_field(*descriptor, &FieldName::parse("control").unwrap())
                .map_err(|error| error.to_string())?
                .clone();
            let control = runtime_text(&control, state, "output control")?;
            let message = runtime_text(&message, state, "output message")?;
            state.emit_local_client_ui_event(
                target,
                LocalClientUiEvent::Output { control, message },
            );
            return Ok(());
        }
        if let Some(target) = routed_client
            && let Value::List(descriptor) = value
        {
            let descriptor = state
                .heap
                .list(*descriptor)
                .map_err(|error| error.to_string())?;
            let keyed = |name: &str| descriptor.get_key(&Value::text(name)).ok().cloned();
            if let Some(kind) = keyed("kind").as_ref().and_then(value_text) {
                if kind == "browse_rsc" {
                    let resource = keyed("resource").unwrap_or(Value::Null);
                    let name = keyed("name")
                        .as_ref()
                        .and_then(value_text)
                        .unwrap_or_default()
                        .to_owned();
                    let path = match resource {
                        Value::File(path) | Value::Text(path) => PathBuf::from(path.as_ref()),
                        _ => PathBuf::new(),
                    };
                    let path = if path.is_absolute() {
                        path
                    } else {
                        match state.project_root.as_ref() {
                            Some(root) => root.join(&path),
                            None => path,
                        }
                    };
                    let bytes = fs::read(&path).unwrap_or_else(|error| {
                        eprintln!(
                            "browse_rsc warning: optional resource {} is unavailable: {error}",
                            path.display()
                        );
                        if path
                            .extension()
                            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
                        {
                            b"{}".to_vec()
                        } else {
                            Vec::new()
                        }
                    });
                    state.emit_local_client_ui_event(
                        target,
                        LocalClientUiEvent::BrowseResource { name, bytes },
                    );
                    return Ok(());
                }
            }
            if let Some(body) = keyed("body") {
                let html = runtime_text(&body, state, "browse body")?;
                let options = keyed("options")
                    .map(|value| runtime_text(&value, state, "browse options"))
                    .transpose()?
                    .unwrap_or_default();
                let window = options
                    .split(';')
                    .find_map(|item| item.trim().strip_prefix("window="))
                    .and_then(|value| value.split('&').next())
                    .unwrap_or_default()
                    .to_owned();
                if !window.is_empty()
                    && let Some(session) = state.client_session_mut(target)
                {
                    session
                        .ensure_browser_window(&window)
                        .map_err(|error| format!("browse window creation failed: {error:?}"))?;
                }
                state.emit_local_client_ui_event(
                    target,
                    LocalClientUiEvent::Browse { window, html },
                );
                return Ok(());
            }
            if let (Some(message), Some(control)) = (keyed("message"), keyed("control")) {
                state.emit_local_client_ui_event(
                    target,
                    LocalClientUiEvent::Output {
                        control: runtime_text(&control, state, "output control")?,
                        message: runtime_text(&message, state, "output message")?,
                    },
                );
                return Ok(());
            }
        }
        let field = FieldName::parse("_dream64_output_events")
            .expect("headless output event field is valid");
        let events = if let Ok(Value::List(events)) = state.heap.datum_field(*target, &field) {
            *events
        } else {
            let events = state.heap.allocate_list();
            state
                .heap
                .set_datum_field(*target, field, Value::List(events))
                .map_err(|error| error.to_string())?;
            events
        };
        state
            .heap
            .list_mut(events)
            .map_err(|error| error.to_string())?
            .add(value.clone());
        return Ok(());
    }
    let Value::Text(_) = target else {
        return Ok(());
    };
    let path = prepare_write_file_path(std::slice::from_ref(target), state, "output")?
        .ok_or_else(|| "output failed to create destination parent".to_owned())?;
    let text = runtime_text(value, state, "output value")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("output failed: {error}"))?;
    writeln!(file, "{text}").map_err(|error| format!("output failed: {error}"))
}

pub(super) fn html_encode(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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

pub(super) fn html_decode(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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
