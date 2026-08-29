//! `md5()`, `rust_g hash_string/hash_file/url_encode/url_decode` and the
//! byte nibble table.

use std::fs;

use dm_value::Value;

use super::{
    ExecutionState, icons::icon_backing_resource, relaxed_resolved_file_path, strict_text,
};
pub(super) fn md5_builtin(arguments: &[Value]) -> Result<Value, String> {
    let Some(Value::Text(text)) = arguments.first() else {
        return Ok(Value::Null);
    };
    Ok(Value::text(format!("{:x}", md5::compute(text.as_bytes()))))
}

pub(super) fn rust_g_hash_string(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let algorithm = strict_text(&arguments[0], state, "hash_string algorithm")?;
    let text = strict_text(&arguments[1], state, "hash_string text")?;
    match algorithm.to_ascii_lowercase().as_str() {
        "md5" => Ok(Value::text(format!("{:x}", md5::compute(text.as_bytes())))),
        algorithm => Err(format!(
            "hash_string algorithm {algorithm:?} is unavailable in the Dream64 host"
        )),
    }
}

pub(super) fn rust_g_hash_file(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let algorithm = strict_text(&arguments[0], state, "hash_file algorithm")?;
    if !algorithm.eq_ignore_ascii_case("md5") {
        return Err(format!(
            "hash_file algorithm {algorithm:?} is unavailable in the Dream64 host"
        ));
    }
    // Native BYOND extension arguments are stringified at the DLL boundary.
    // In particular, an `/icon` backed by a resource is passed to rust-g as
    // that resource path. Preserve the datum distinction internally, then
    // unwrap it for this file-taking external call.
    let icon_argument = matches!(&arguments[1], Value::Datum(_));
    let path_argument = match &arguments[1] {
        Value::Datum(_) => icon_backing_resource(&arguments[1], state, 0)?,
        value => value.clone(),
    };
    let path = relaxed_resolved_file_path(&[path_argument], state, "hash_file path")?;
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        // Headless icon composition preserves the intended backing resource
        // path but does not rasterize generated spritesheets to disk. Keep
        // cache keys stable for those virtual icon resources.
        Err(error) if icon_argument && error.kind() == std::io::ErrorKind::NotFound => {
            path.to_string_lossy().as_bytes().to_vec()
        }
        Err(error) => {
            return Err(format!(
                "hash_file failed to read '{}': {error}",
                path.display()
            ));
        }
    };
    Ok(Value::text(format!("{:x}", md5::compute(bytes))))
}

pub(super) fn rust_g_url_encode(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "url_encode input")?;
    let mut encoded = String::with_capacity(text.len());
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for byte in text.bytes() {
        match byte {
            b' ' => encoded.push('+'),
            b'*' | b'-' | b'.' | b'0'..=b'9' | b'A'..=b'Z' | b'_' | b'a'..=b'z' => {
                encoded.push(char::from(byte));
            }
            _ => {
                encoded.push('%');
                encoded.push(char::from(HEX[usize::from(byte >> 4)]));
                encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
            }
        }
    }
    Ok(Value::text(encoded))
}

pub(super) fn rust_g_url_decode(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "url_decode input")?;
    let source = text.as_bytes();
    let mut decoded = Vec::with_capacity(source.len());
    let mut index = 0;
    while index < source.len() {
        if source[index] == b'+' {
            decoded.push(b' ');
            index += 1;
            continue;
        }
        if source[index] == b'%'
            && index + 2 < source.len()
            && let (Some(high), Some(low)) =
                (hex_nibble(source[index + 1]), hex_nibble(source[index + 2]))
        {
            decoded.push((high << 4) | low);
            index += 3;
            continue;
        }
        decoded.push(source[index]);
        index += 1;
    }
    Ok(Value::text(String::from_utf8_lossy(&decoded).into_owned()))
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
