//! Text-to-number chores, `list2params`/`params2list`, and BYOND form
//! encoding.

use std::fmt::Write;

use dm_value::Value;

use super::{ExecutionState, number, runtime_text, strict_text};
pub(super) fn lentext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "lentext")?;
    Ok(Value::number(text.len() as f32))
}

pub(super) fn sorttext(
    arguments: &[Value],
    state: &ExecutionState,
    exact: bool,
) -> Result<Value, String> {
    if arguments.len() < 2 {
        return Ok(Value::number(0.0));
    }
    let values = arguments
        .iter()
        // BYOND sorttext is a comparator over each value's text
        // representation. It accepts null (""), numbers, type paths, and
        // datums; tg/Monk relies on this while sorting associative type
        // catalogs whose optional display key can be null.
        .map(|value| runtime_text(value, state, "sorttext"))
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

pub(super) fn num2text(arguments: &[Value]) -> Result<Value, String> {
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

pub(super) fn list2params(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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
        let encoded_key = form_encode(&key_text);
        if let Ok(associated) = list.get_key(key) {
            let value_text = runtime_text(associated, state, "list2params value")?;
            pairs.push(format!("{encoded_key}={}", form_encode(&value_text)));
        } else {
            pairs.push(encoded_key);
        }
    }
    Ok(Value::text(pairs.join("&")))
}

pub(crate) fn params2list(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
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
