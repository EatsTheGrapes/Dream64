//! Native DMI metadata, `icon`/`iconStates`/`_dream64_icon_swap_color`
//! construction, and resource-baked icon lookup for streaming engines.

use std::collections::HashMap;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use dm_value::{FieldName, Value};

use super::generator::resource_datum_builtin;
use super::{ExecutionState, relaxed_resolved_file_path};

#[derive(Clone, Debug)]
pub(crate) struct DmiMetadata {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) states: Vec<DmiState>,
    pub(crate) error: Option<String>,
}

impl DmiMetadata {
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "width": self.width,
            "height": self.height,
            "states": self.states.iter().map(DmiState::to_json).collect::<Vec<_>>(),
            "headless_error": self.error,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DmiState {
    pub(crate) name: String,
    dirs: u32,
    frames: u32,
    delay: Vec<f64>,
    loop_value: i64,
    rewind: i64,
    movement: i64,
    hotspot: Option<Vec<i64>>,
}

impl DmiState {
    fn new(name: String) -> Self {
        Self {
            name,
            dirs: 1,
            frames: 1,
            delay: Vec::new(),
            loop_value: 0,
            rewind: 0,
            movement: 0,
            hotspot: None,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "dirs": self.dirs,
            "frames": self.frames,
            "delay": self.delay,
            "loop": self.loop_value,
            "rewind": self.rewind,
            "movement": self.movement,
            "hotspot": self.hotspot,
        })
    }
}

#[derive(Clone)]
struct CachedDmiMetadata {
    len: u64,
    modified: Option<SystemTime>,
    metadata: DmiMetadata,
}

const MAX_DMI_METADATA_CACHE_ENTRIES: usize = 4_096;
static DMI_METADATA_CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedDmiMetadata>>> = OnceLock::new();
#[cfg(test)]
pub(crate) static DMI_METADATA_PHYSICAL_READS: OnceLock<Mutex<HashMap<PathBuf, u64>>> =
    OnceLock::new();

pub(crate) fn read_dmi_metadata(path: &Path) -> Result<DmiMetadata, String> {
    let file = fs::metadata(path).map_err(|error| error.to_string())?;
    let len = file.len();
    let modified = file.modified().ok();
    let cache = DMI_METADATA_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(cache) = cache.lock()
        && let Some(entry) = cache.get(path)
        && entry.len == len
        && entry.modified == modified
    {
        return Ok(entry.metadata.clone());
    }

    let metadata = read_dmi_metadata_uncached(path)?;
    if let Ok(mut cache) = cache.lock() {
        if cache.len() >= MAX_DMI_METADATA_CACHE_ENTRIES && !cache.contains_key(path) {
            cache.clear();
        }
        cache.insert(
            path.to_path_buf(),
            CachedDmiMetadata {
                len,
                modified,
                metadata: metadata.clone(),
            },
        );
    }
    Ok(metadata)
}

fn read_dmi_metadata_uncached(path: &std::path::Path) -> Result<DmiMetadata, String> {
    #[cfg(test)]
    if let Ok(mut reads) = DMI_METADATA_PHYSICAL_READS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        *reads.entry(path.to_path_buf()).or_default() += 1;
    }
    let png = fs::read(path).map_err(|error| error.to_string())?;
    if !png.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("resource is not a PNG-backed DMI".to_owned());
    }
    let mut cursor = 8usize;
    let mut image_width = None;
    let mut image_height = None;
    let mut description = None;
    while cursor.checked_add(12).is_some_and(|end| end <= png.len()) {
        let length = u32::from_be_bytes(
            png[cursor..cursor + 4]
                .try_into()
                .map_err(|_| "invalid PNG chunk length")?,
        ) as usize;
        let chunk_type = &png[cursor + 4..cursor + 8];
        let data_start = cursor + 8;
        let data_end = data_start
            .checked_add(length)
            .ok_or("PNG chunk length overflow")?;
        let chunk_end = data_end.checked_add(4).ok_or("PNG chunk CRC overflow")?;
        if chunk_end > png.len() {
            return Err("truncated PNG chunk".to_owned());
        }
        let data = &png[data_start..data_end];
        match chunk_type {
            b"IHDR" if data.len() >= 8 => {
                image_width = Some(u32::from_be_bytes(data[0..4].try_into().unwrap()));
                image_height = Some(u32::from_be_bytes(data[4..8].try_into().unwrap()));
            }
            b"tEXt" => {
                if let Some(separator) = data.iter().position(|byte| *byte == 0)
                    && &data[..separator] == b"Description"
                {
                    description = Some(
                        String::from_utf8(data[separator + 1..].to_vec())
                            .map_err(|error| error.to_string())?,
                    );
                }
            }
            b"zTXt" => {
                if let Some(separator) = data.iter().position(|byte| *byte == 0)
                    && &data[..separator] == b"Description"
                {
                    let method = *data
                        .get(separator + 1)
                        .ok_or("DMI zTXt chunk lacks compression method")?;
                    if method != 0 {
                        return Err(format!("unsupported DMI zTXt compression method {method}"));
                    }
                    let compressed = data
                        .get(separator + 2..)
                        .ok_or("DMI zTXt chunk lacks compressed data")?;
                    let mut decoder = flate2::read::ZlibDecoder::new(compressed);
                    let mut decoded = String::new();
                    decoder
                        .read_to_string(&mut decoded)
                        .map_err(|error| error.to_string())?;
                    description = Some(decoded);
                }
            }
            _ => {}
        }
        cursor = chunk_end;
        if description.is_some() && image_width.is_some() {
            break;
        }
    }
    let image_width = image_width.ok_or("PNG is missing IHDR width")?;
    let image_height = image_height.ok_or("PNG is missing IHDR height")?;
    match description {
        Some(description) => parse_dmi_description(&description, image_width, image_height),
        None => Ok(DmiMetadata {
            width: image_width,
            height: image_height,
            states: vec![DmiState::new(String::new())],
            error: None,
        }),
    }
}

pub(crate) fn icon_states_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let mut resource = match arguments.first().cloned().unwrap_or(Value::Null) {
        Value::Datum(datum) => super::datum_field_or_initial(
            state,
            datum,
            &FieldName::parse("icon").expect("icon field name is valid"),
        )
        .map_err(|error| error.to_string())?,
        value => value,
    };
    if let Value::Datum(_) = resource {
        resource = icon_backing_resource(&resource, state, 0)?;
    }
    let requested = match resource {
        Value::File(path) => path.to_string(),
        Value::Text(path) => path.to_string(),
        Value::Null => {
            return Err("icon_states resource requires text, received null".to_owned());
        }
        value => {
            return Err(format!(
                "icon_states resource requires text, received {value}"
            ));
        }
    };
    let resolved = relaxed_resolved_file_path(
        &[Value::text(requested.clone())],
        state,
        "icon_states resource",
    )?;
    let metadata = read_dmi_metadata(&resolved).map_err(|error| {
        format!(
            "icon_states failed for resource {requested:?} resolved to '{}': {error}",
            resolved.display()
        )
    })?;
    // BYOND's `mode` argument: 0 (default) yields every state, 1 restricts the
    // result to movement states. DMI metadata carries the `movement` flag per
    // state, so both modes are honoured exactly.
    let movement_only = arguments
        .get(1)
        .and_then(Value::as_number)
        .is_some_and(|mode| mode as i64 == 1);
    let list = state.heap_mut().allocate_list();
    let values = state
        .heap_mut()
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for icon_state in metadata.states {
        if movement_only && icon_state.movement == 0 {
            continue;
        }
        values.add(Value::text(icon_state.name));
    }
    Ok(Value::List(list))
}

fn parse_dmi_description(
    description: &str,
    image_width: u32,
    image_height: u32,
) -> Result<DmiMetadata, String> {
    let mut metadata = DmiMetadata {
        width: image_width,
        height: image_height,
        states: Vec::new(),
        error: None,
    };
    let mut state = None;
    for raw_line in description.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("invalid DMI metadata line {line:?}"));
        };
        let key = key.trim();
        let value = value.trim();
        match key {
            "version" => {}
            "width" => {
                metadata.width = value
                    .parse()
                    .map_err(|error| format!("invalid DMI width {value:?}: {error}"))?;
            }
            "height" => {
                metadata.height = value
                    .parse()
                    .map_err(|error| format!("invalid DMI height {value:?}: {error}"))?;
            }
            "state" => {
                if let Some(previous) = state.take() {
                    metadata.states.push(previous);
                }
                let name = serde_json::from_str::<String>(value)
                    .unwrap_or_else(|_| value.trim_matches('"').to_owned());
                state = Some(DmiState::new(name));
            }
            "dirs" => {
                if let Some(state) = state.as_mut() {
                    state.dirs = value
                        .parse()
                        .map_err(|error| format!("invalid DMI dirs {value:?}: {error}"))?;
                }
            }
            "frames" => {
                if let Some(state) = state.as_mut() {
                    state.frames = value
                        .parse()
                        .map_err(|error| format!("invalid DMI frames {value:?}: {error}"))?;
                }
            }
            "delay" => {
                if let Some(state) = state.as_mut() {
                    state.delay = value
                        .split(',')
                        .map(str::trim)
                        .map(str::parse)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| format!("invalid DMI delay {value:?}: {error}"))?;
                }
            }
            "loop" | "rewind" | "movement" => {
                if let Some(state) = state.as_mut() {
                    let parsed = value
                        .parse()
                        .map_err(|error| format!("invalid DMI {key} {value:?}: {error}"))?;
                    match key {
                        "loop" => state.loop_value = parsed,
                        "rewind" => state.rewind = parsed,
                        _ => state.movement = parsed,
                    }
                }
            }
            "hotspot" => {
                if let Some(state) = state.as_mut() {
                    state.hotspot = Some(
                        value
                            .split(',')
                            .map(str::trim)
                            .map(str::parse)
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|error| format!("invalid DMI hotspot {value:?}: {error}"))?,
                    );
                }
            }
            _ => return Err(format!("unsupported DMI metadata key {key:?}")),
        }
    }
    if let Some(state) = state {
        metadata.states.push(state);
    }
    Ok(metadata)
}
pub(crate) fn icon_swap_color_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let Value::Datum(icon) = arguments[0] else {
        return Err("icon.SwapColor requires an /icon receiver".to_owned());
    };
    if !super::is_icon_datum(icon, state.heap()) {
        return Err("icon.SwapColor requires an /icon receiver".to_owned());
    }
    super::execute_icon_method(icon, "SwapColor", &arguments[1..], state.heap_mut())
}

/// Constructs BYOND's mutable `/icon` value.
///
/// An existing `/icon` is a copy-constructor input, not the backing resource
/// stored in the new icon's `icon` field. `OpenDream`'s `DreamObjectIcon`
/// mirrors BYOND by copying its complete `DreamIcon` here. This is observable
/// in tg-derived `getFlatIcon()`, which starts every render with
/// `flat_template = icon(file); flat = icon(flat_template)` and then mutates
/// `flat` independently.
pub(crate) fn icon_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if let Some(Value::Datum(source)) = arguments.first()
        && super::is_icon_datum(*source, &state.heap)
    {
        return super::clone_icon_datum(*source, &mut state.heap).map(Value::Datum);
    }

    let icon = resource_datum_builtin(
        "/icon",
        &["icon", "icon_state", "dir", "frame", "moving"],
        arguments,
        state,
    )?;

    // A BYOND /icon owns the dimensions of the selected DMI frame, not the
    // engine's 32x32 fallback. Large canvas DMIs (Monkestation's holomap is
    // 480x480) immediately observe this through Width()/Height(). Keep the
    // constructor permissive for synthetic/missing headless resources, but
    // seed exact metadata whenever the backing resource is available.
    if let (Value::Datum(icon), Some(Value::File(_) | Value::Text(_))) = (&icon, arguments.first())
        && let Ok(resolved) =
            relaxed_resolved_file_path(&arguments[..1], state, "icon constructor resource")
        && let Ok(metadata) = read_dmi_metadata(&resolved)
    {
        state
            .heap_mut()
            .set_datum_field(
                *icon,
                FieldName::parse("_dream64_width").expect("internal icon width is valid"),
                Value::number(metadata.width as f32),
            )
            .map_err(|error| error.to_string())?;
        state
            .heap_mut()
            .set_datum_field(
                *icon,
                FieldName::parse("_dream64_height").expect("internal icon height is valid"),
                Value::number(metadata.height as f32),
            )
            .map_err(|error| error.to_string())?;
    }

    Ok(icon)
}

pub(crate) fn fcopy_rsc(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(value) = arguments.first() else {
        return Ok(Value::Null);
    };
    match value {
        Value::File(path) | Value::Text(path) => Ok(Value::file(path.clone())),
        Value::Null => Ok(Value::Null),
        Value::Datum(_) => icon_backing_resource(value, state, 0),
        value => Err(format!(
            "fcopy_rsc requires a file, path, or icon, received {value}"
        )),
    }
}

/// `OpenDream`, matching BYOND, keeps `/icon` objects distinct from resources:
/// `isfile(icon)` is false, while `fcopy_rsc(icon)` materializes an icon
/// resource. Dream64's headless renderer retains the constructor's backing
/// resource instead of rasterizing pixels, so unwrap that backing resource
/// (including icons cloned from other icons) into the first-class `File`
/// value used by filesystem/resource builtins.
pub(crate) fn icon_backing_resource(
    value: &Value,
    state: &ExecutionState,
    depth: usize,
) -> Result<Value, String> {
    if depth >= 64 {
        return Err("fcopy_rsc encountered a cyclic icon resource".to_owned());
    }
    let Value::Datum(icon) = value else {
        return Err(format!("fcopy_rsc requires an icon, received {value}"));
    };
    let datum = state.heap.datum(*icon).map_err(|error| error.to_string())?;
    let path = datum.type_path().as_str();
    if path != "/icon" && !path.starts_with("/icon/") {
        return Err(format!("fcopy_rsc requires an icon, received {value}"));
    }
    let field = FieldName::parse("icon").expect("built-in icon field is valid");
    match datum.field(&field) {
        Ok(Value::File(path) | Value::Text(path)) => Ok(Value::file(path.clone())),
        Ok(Value::Datum(backing)) => {
            icon_backing_resource(&Value::Datum(*backing), state, depth + 1)
        }
        Ok(Value::Null) | Err(_) => Ok(Value::Null),
        Ok(value) => Err(format!(
            "fcopy_rsc icon has an unsupported backing resource {value}"
        )),
    }
}
