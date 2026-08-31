//! `world` profile/config introspection and the local listen port bridge.

use dm_value::{FieldName, Value};

use super::{ExecutionState, strict_text, value_text};
pub(super) fn world_profile(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
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

pub(super) fn world_get_config(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
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

pub(super) fn world_set_config(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
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

pub(super) fn world_open_port(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
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
