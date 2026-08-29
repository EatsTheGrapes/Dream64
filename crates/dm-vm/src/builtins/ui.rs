//! Headless client/browser/skin transport (`browse`, `winset`, `winget`, ...),
//! `regex_quote`, `floor_multiple`, and the local-client session resolver.

use dm_dmf::UiCommand;

use dm_value::{DatumId, FieldName, Value};

use crate::LocalClientUiEvent;

use super::{ExecutionState, runtime_text, truthy, value_text};
pub(super) fn regex_quote(
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

pub(super) fn headless_browse(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
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

pub(super) fn headless_transfer(
    kind: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let descriptor = state.heap.allocate_list();
    state.mark_associative_list(descriptor);
    let list = state
        .heap
        .list_mut(descriptor)
        .map_err(|error| error.to_string())?;
    list.set_key(Value::text("kind"), Value::text(kind));
    list.set_key(
        Value::text("resource"),
        arguments.first().cloned().unwrap_or(Value::Null),
    );
    list.set_key(
        Value::text("name"),
        arguments.get(1).cloned().unwrap_or(Value::Null),
    );
    Ok(Value::List(descriptor))
}

pub(crate) fn local_client_for_value(state: &ExecutionState, value: &Value) -> Option<DatumId> {
    let Value::Datum(datum) = value else {
        return None;
    };
    if state.client_session(*datum).is_some() {
        return Some(*datum);
    }
    let client = FieldName::parse("client").expect("engine client field");
    let Value::Datum(client) = state.heap.datum_field(*datum, &client).ok()? else {
        return None;
    };
    state.client_session(*client).is_some().then_some(*client)
}

pub(super) fn headless_link(
    arguments: &[Value],
    state: &mut ExecutionState,
    usr: &Value,
) -> Result<Value, String> {
    let value = arguments.first().cloned().unwrap_or(Value::Null);
    if let Some(url) = value_text(&value)
        && let Some(client) = local_client_for_value(state, usr)
    {
        state.emit_local_client_ui_event(
            client,
            LocalClientUiEvent::Link {
                url: url.to_owned(),
            },
        );
    }
    Ok(value)
}

pub(super) fn headless_winset(
    arguments: &[Value],
    state: &mut ExecutionState,
    usr: &Value,
) -> Result<Value, String> {
    // Preserve UI state on an explicit headless /client even before a native
    // session is attached. Session lookup is intentionally stricter because
    // mob/client forwarding only makes sense for a registered local client.
    let explicit_headless_client = arguments.first().and_then(|value| {
        let Value::Datum(datum) = value else {
            return None;
        };
        state
            .heap
            .datum(*datum)
            .ok()
            .is_some_and(|value| {
                let path = value.type_path().as_str();
                path == "/client" || path.starts_with("/client/")
            })
            .then_some(*datum)
    });
    let Some(client) = explicit_headless_client.or_else(|| {
        arguments
            .first()
            .and_then(|value| local_client_for_value(state, value))
            .or_else(|| local_client_for_value(state, usr))
    }) else {
        // BYOND accepts null when no client is available; a headless server
        // has no window to mutate in that case.
        return Ok(Value::Null);
    };
    if let (Some(control), Some(parameters), Some(session)) = (
        arguments.get(1).and_then(value_text),
        arguments.get(2).and_then(value_text),
        state.client_session_mut(client),
    ) {
        session
            .apply_command(UiCommand::WinSet {
                control: control.to_owned(),
                parameters: parameters.to_owned(),
            })
            .map_err(|error| format!("client UI winset failed: {error:?}"))?;
        state.emit_local_client_ui_event(
            client,
            LocalClientUiEvent::Winset {
                control: control.to_owned(),
                parameters: parameters.to_owned(),
            },
        );
        return Ok(Value::Null);
    }
    let field = FieldName::parse("_dream64_winset").expect("headless UI field is valid");
    let settings = if let Ok(Value::List(settings)) = state.heap.datum_field(client, &field) {
        *settings
    } else {
        let settings = state.heap.allocate_list();
        state
            .heap
            .set_datum_field(client, field, Value::List(settings))
            .map_err(|error| error.to_string())?;
        settings
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

pub(super) fn headless_winshow(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let Some(Value::Datum(client)) = arguments.first() else {
        return Ok(Value::Null);
    };
    if let (Some(control), Some(session)) = (
        arguments.get(1).and_then(value_text),
        state.client_session_mut(*client),
    ) {
        session
            .apply_command(UiCommand::WinShow {
                control: control.to_owned(),
                visible: arguments.get(2).is_none_or(truthy),
            })
            .map_err(|error| format!("client UI winshow failed: {error:?}"))?;
        return Ok(Value::Null);
    }
    let field = FieldName::parse("_dream64_winshow").expect("headless UI field");
    let settings = if let Ok(Value::List(list)) = state.heap.datum_field(*client, &field) {
        *list
    } else {
        let list = state.heap.allocate_list();
        state.mark_associative_list(list);
        state
            .heap
            .set_datum_field(*client, field, Value::List(list))
            .map_err(|e| e.to_string())?;
        list
    };
    state
        .heap
        .list_mut(settings)
        .map_err(|e| e.to_string())?
        .set_key(
            arguments.get(1).cloned().unwrap_or(Value::Null),
            arguments
                .get(2)
                .cloned()
                .unwrap_or_else(|| Value::number(1.0)),
        );
    Ok(Value::Null)
}

pub(super) fn headless_winclone(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let [Value::Datum(client), source, destination] = arguments else {
        return Ok(Value::number(0.0));
    };
    if let (Some(source), Some(destination), Some(session)) = (
        value_text(source),
        value_text(destination),
        state.client_session_mut(*client),
    ) {
        session
            .apply_command(UiCommand::WinClone {
                source: source.to_owned(),
                destination: destination.to_owned(),
            })
            .map_err(|error| format!("client UI winclone failed: {error:?}"))?;
        return Ok(Value::number(1.0));
    }
    let field = FieldName::parse("_dream64_winset").expect("headless UI field");
    let Some(Value::List(settings)) = state.heap.datum_field(*client, &field).ok().cloned() else {
        return Ok(Value::number(0.0));
    };
    let value = state
        .heap
        .list(settings)
        .map_err(|e| e.to_string())?
        .get_key(source)
        .ok()
        .cloned();
    let Some(value) = value else {
        return Ok(Value::number(0.0));
    };
    state
        .heap
        .list_mut(settings)
        .map_err(|e| e.to_string())?
        .set_key(destination.clone(), value);
    Ok(Value::number(1.0))
}

pub(super) fn headless_winget(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let (client, control, property) = match arguments {
        [Value::Datum(client), control, property] => (*client, control, property),
        _ => return Ok(Value::text("")),
    };
    if let (Some(control), Some(property), Some(session)) = (
        value_text(control),
        value_text(property),
        state.client_session(client),
    ) {
        return session
            .ui()
            .winget(control, property)
            .map(Value::text)
            .map_err(|error| format!("client UI winget failed: {error:?}"));
    }
    let Some(Value::List(settings)) = state
        .heap
        .datum_field(
            client,
            &FieldName::parse("_dream64_winset").expect("headless UI field is valid"),
        )
        .ok()
    else {
        return Ok(Value::text(""));
    };
    let Some(control) = value_text(control) else {
        return Ok(Value::text(""));
    };
    let Some(property) = value_text(property) else {
        return Ok(Value::text(""));
    };
    let settings = state
        .heap
        .list(*settings)
        .map_err(|error| error.to_string())?;
    let Ok(value) = settings.get_key(&Value::text(control)) else {
        return Ok(Value::text(""));
    };
    let Some(parameters) = value_text(value) else {
        return Ok(Value::text(""));
    };
    let value = parameters.split(';').find_map(|entry| {
        let (name, value) = entry.split_once('=')?;
        (name.trim() == property).then(|| value.trim().to_owned())
    });
    Ok(Value::text(value.unwrap_or_default()))
}

pub(super) fn headless_winexists(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let Some(Value::Datum(client)) = arguments.first() else {
        return Ok(Value::number(0.0));
    };
    if let (Some(control), Some(session)) = (
        arguments.get(1).and_then(value_text),
        state.client_session(*client),
    ) {
        // BYOND returns the matching control's type, not merely a boolean.
        // Monkestation's media player relies on `winexists(...) == "BROWSER"`
        // to select the embedded-browser output address used for lobby music.
        return Ok(Value::text(session.ui().winexists_type(control)));
    }
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
pub(super) fn headless_alert(arguments: &[Value]) -> Result<Value, String> {
    let explicit_usr =
        arguments.len() >= 4 && matches!(arguments.first(), Some(Value::Datum(_) | Value::Null));
    let button = arguments
        .get(if explicit_usr { 3 } else { 2 })
        .filter(|value| !matches!(value, Value::Null))
        .cloned()
        .unwrap_or_else(|| Value::text("Ok"));
    Ok(button)
}

pub(super) fn floor_multiple(arguments: &[Value]) -> Result<Value, String> {
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
