//! Native DMI metadata, `icon`/`iconStates`/`_dream64_icon_swap_color`
//! construction, and resource-baked icon lookup for streaming engines.

use std::collections::HashMap;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use dm_value::{DatumId, FieldName, Value, ValueHeap};

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
    /// Monotonic tick of the last read that served or filled this entry. Used
    /// for bounded least-recently-used eviction instead of clearing the whole
    /// cache when it fills.
    last_used: u64,
    metadata: DmiMetadata,
}

// A `DmiMetadata` is small, and SS13 references far more than a few thousand
// distinct DMIs (`SSgreyscale_previews` alone `icon()`s thousands of preview
// sheets). Keep the cap generous and evict one entry at a time.
const MAX_DMI_METADATA_CACHE_ENTRIES: usize = 16_384;
static DMI_METADATA_CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedDmiMetadata>>> = OnceLock::new();
static DMI_METADATA_CLOCK: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
pub(crate) static DMI_METADATA_PHYSICAL_READS: OnceLock<Mutex<HashMap<PathBuf, u64>>> =
    OnceLock::new();

/// Inserts `entry` for `path`, first making room with bounded least-recently-used
/// eviction: while the cache is at capacity and this is a new key, drop the
/// single entry with the oldest `last_used` tick. This replaces the previous
/// clear-the-whole-cache-when-full behaviour, which forced every subsequent
/// `icon()` / `IconStates()` to re-decode a PNG once a boot referenced more than
/// the cap's worth of distinct DMIs.
fn store_dmi_metadata(
    cache: &mut HashMap<PathBuf, CachedDmiMetadata>,
    path: &Path,
    entry: CachedDmiMetadata,
) {
    while cache.len() >= MAX_DMI_METADATA_CACHE_ENTRIES && !cache.contains_key(path) {
        let Some(victim) = cache
            .iter()
            .min_by_key(|(_, cached)| cached.last_used)
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        cache.remove(&victim);
    }
    cache.insert(path.to_path_buf(), entry);
}

pub(crate) fn read_dmi_metadata(path: &Path) -> Result<DmiMetadata, String> {
    let file = fs::metadata(path).map_err(|error| error.to_string())?;
    let len = file.len();
    let modified = file.modified().ok();
    let cache = DMI_METADATA_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut cache) = cache.lock()
        && let Some(entry) = cache.get_mut(path)
        && entry.len == len
        && entry.modified == modified
    {
        entry.last_used = DMI_METADATA_CLOCK.fetch_add(1, Ordering::Relaxed);
        return Ok(entry.metadata.clone());
    }

    let metadata = read_dmi_metadata_uncached(path)?;
    if let Ok(mut cache) = cache.lock() {
        store_dmi_metadata(
            &mut cache,
            path,
            CachedDmiMetadata {
                len,
                modified,
                last_used: DMI_METADATA_CLOCK.fetch_add(1, Ordering::Relaxed),
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

/// Decoded-DMI cache shared by GAGS compositing and `/icon` materialisation.
/// Asset generation runs a handful of source DMIs through many operations; a
/// path+mtime keyed cache turns each repeat into an `Arc` clone.
type DmiBitmapCache = HashMap<PathBuf, (Option<SystemTime>, Arc<dm_icon::IconBitmap>)>;
static DMI_BITMAP_CACHE: OnceLock<Mutex<DmiBitmapCache>> = OnceLock::new();

/// Read and decode a DMI, returning a shared cached copy when the file is
/// unchanged since the last decode.
pub(crate) fn load_dmi_bitmap_cached(path: &Path) -> Result<Arc<dm_icon::IconBitmap>, String> {
    let modified = fs::metadata(path)
        .ok()
        .and_then(|meta| meta.modified().ok());
    let cache = DMI_BITMAP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(cache) = cache.lock()
        && let Some((cached_mtime, bitmap)) = cache.get(path)
        && *cached_mtime == modified
    {
        return Ok(Arc::clone(bitmap));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read DMI '{}': {error}", path.display()))?;
    let bitmap = Arc::new(
        dm_icon::IconBitmap::from_dmi_bytes(&bytes)
            .map_err(|error| format!("cannot decode DMI '{}': {error}", path.display()))?,
    );
    if let Ok(mut cache) = cache.lock() {
        if cache.len() >= 128 {
            cache.clear();
        }
        cache.insert(path.to_path_buf(), (modified, Arc::clone(&bitmap)));
    }
    Ok(bitmap)
}

fn icon_field<'a>(heap: &'a ValueHeap, icon: DatumId, name: &str) -> Option<&'a Value> {
    let field = FieldName::parse(name).ok()?;
    heap.datum_field(icon, &field).ok()
}

/// Length of an icon's `_dream64_icon_journal` (0 when never mutated).
pub(crate) fn icon_journal_len(heap: &ValueHeap, icon: DatumId) -> usize {
    match icon_field(heap, icon, "_dream64_icon_journal") {
        Some(Value::List(list)) => match heap.list(*list) {
            Ok(entries) => entries.len(),
            Err(_) => 0,
        },
        _ => 0,
    }
}

fn journal_color(value: &Value) -> Option<dm_icon::Rgba> {
    match value {
        Value::Text(text) => dm_icon::parse_color(text),
        _ => None,
    }
}

/// Build a composited [`dm_icon::IconBitmap`] for `icon` by decoding its backing
/// DMI and replaying every recorded `_dream64_icon_journal` entry in order.
///
/// Returns `Ok(None)` when the icon has no resolvable backing DMI (a synthetic
/// headless icon) so callers can fall back to their existing behaviour.
pub(crate) fn materialize_icon_bitmap(
    icon: DatumId,
    state: &ExecutionState,
    depth: usize,
) -> Result<Option<dm_icon::IconBitmap>, String> {
    if depth >= 32 {
        return Err("icon materialization recursed too deeply".to_owned());
    }
    let heap = &state.heap;
    if !super::is_icon_datum(icon, heap) {
        return Ok(None);
    }

    // Resolve the base bitmap from the backing resource.
    let mut bitmap = match icon_field(heap, icon, "icon").cloned() {
        Some(Value::File(path) | Value::Text(path)) => {
            let resolved = relaxed_resolved_file_path(
                &[Value::text(path.to_string())],
                state,
                "icon materialization resource",
            )?;
            load_dmi_bitmap_cached(&resolved)?.as_ref().clone()
        }
        Some(Value::Datum(backing)) => match materialize_icon_bitmap(backing, state, depth + 1)? {
            Some(bitmap) => bitmap,
            None => return Ok(None),
        },
        _ => return Ok(None),
    };

    // `icon(file, "state")` narrows the icon to a single state.
    if let Some(Value::Text(state_name)) = icon_field(heap, icon, "icon_state")
        && !state_name.is_empty()
    {
        bitmap = bitmap.select_state(state_name);
    }

    let Some(Value::List(journal)) = icon_field(heap, icon, "_dream64_icon_journal").cloned()
    else {
        return Ok(Some(bitmap));
    };
    let entries = heap.list(journal).map_err(|error| error.to_string())?.len();
    for index in 1..=entries {
        let Value::List(op) = heap
            .list(journal)
            .map_err(|error| error.to_string())?
            .get(index)
            .map_err(|error| error.to_string())?
            .clone()
        else {
            continue;
        };
        let op = heap.list(op).map_err(|error| error.to_string())?;
        let entry: Vec<Value> = (1..=op.len())
            .filter_map(|i| op.get(i).ok().cloned())
            .collect();
        apply_journal_entry(&mut bitmap, &entry, state, depth)?;
    }
    Ok(Some(bitmap))
}

/// Apply one `_dream64_icon_journal` entry (`[method, args...]`) to `bitmap`.
fn apply_journal_entry(
    bitmap: &mut dm_icon::IconBitmap,
    entry: &[Value],
    state: &ExecutionState,
    depth: usize,
) -> Result<(), String> {
    let Some(Value::Text(method)) = entry.first() else {
        return Ok(());
    };
    let method = method.to_string();
    // `arg(1)` is the first argument after the method name.
    let arg = |i: usize| entry.get(i);
    let num = |i: usize| arg(i).and_then(Value::as_number);
    match method.as_str() {
        "Scale" => {
            if let (Some(w), Some(h)) = (num(1), num(2).or_else(|| num(1))) {
                bitmap.scale(w.max(1.0) as u32, h.max(1.0) as u32);
            }
        }
        "Crop" => {
            if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (num(1), num(2), num(3), num(4)) {
                bitmap.crop(x1 as i32, y1 as i32, x2 as i32, y2 as i32);
            }
        }
        "Flip" => {
            if let Some(dir) = num(1) {
                bitmap.flip(dir as i64);
            }
        }
        "Turn" => {
            if let Some(angle) = num(1) {
                bitmap.turn(f64::from(angle));
            }
        }
        "Shift" => {
            if let (Some(dir), Some(offset)) = (num(1), num(2)) {
                let wrap = arg(3).is_some_and(super::truthy);
                bitmap.shift(dir as i64, offset as i32, wrap);
            }
        }
        "SwapColor" => {
            if let (Some(old), Some(new)) = (
                arg(1).and_then(journal_color),
                arg(2).and_then(journal_color),
            ) {
                bitmap.swap_color(old, new);
            }
        }
        "DrawBox" => {
            if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (num(2), num(3), num(4), num(5)) {
                let color = arg(1).and_then(journal_color);
                bitmap.draw_box(color, x1 as i32, y1 as i32, x2 as i32, y2 as i32);
            }
        }
        "MapColors" => {
            let matrix: Vec<f32> = entry[1..].iter().filter_map(Value::as_number).collect();
            bitmap.map_colors(&matrix);
        }
        "Blend" => {
            let mode = num(2)
                .and_then(|m| dm_icon::BlendMode::from_byond(m as i64))
                .unwrap_or(dm_icon::BlendMode::Overlay);
            let x = num(3).unwrap_or(1.0) as i32;
            let y = num(4).unwrap_or(1.0) as i32;
            match arg(1) {
                Some(Value::Text(color)) => {
                    if let Some(rgba) = dm_icon::parse_color(color) {
                        bitmap.blend_color(rgba, mode);
                    }
                }
                Some(Value::Datum(other)) => {
                    if let Some(overlay) = materialize_icon_bitmap(*other, state, depth + 1)? {
                        bitmap.blend_icon(&overlay, mode, x, y);
                    }
                }
                _ => {}
            }
        }
        "Insert" => {
            if let Some(Value::Datum(other)) = arg(1)
                && let Some(piece) = materialize_icon_bitmap(*other, state, depth + 1)?
            {
                let state_name = match arg(2) {
                    Some(Value::Text(name)) => name.to_string(),
                    _ => String::new(),
                };
                bitmap.insert(&piece, &state_name);
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod dmi_cache_tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::{
        CachedDmiMetadata, DMI_METADATA_PHYSICAL_READS, DmiMetadata,
        MAX_DMI_METADATA_CACHE_ENTRIES, read_dmi_metadata, store_dmi_metadata,
    };

    fn synthetic_entry(last_used: u64) -> CachedDmiMetadata {
        CachedDmiMetadata {
            len: 0,
            modified: None,
            last_used,
            metadata: DmiMetadata {
                width: 0,
                height: 0,
                states: Vec::new(),
                error: None,
            },
        }
    }

    fn minimal_dmi(width: u32, height: u32) -> Vec<u8> {
        // PNG signature + IHDR + IEND, no BYOND `Description`.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut push_chunk = |kind: &[u8; 4], data: &[u8]| {
            png.extend_from_slice(&(u32::try_from(data.len()).unwrap()).to_be_bytes());
            png.extend_from_slice(kind);
            png.extend_from_slice(data);
            png.extend_from_slice(&[0; 4]);
        };
        let mut header = Vec::new();
        header.extend_from_slice(&width.to_be_bytes());
        header.extend_from_slice(&height.to_be_bytes());
        header.extend_from_slice(&[8, 6, 0, 0, 0]);
        push_chunk(b"IHDR", &header);
        push_chunk(b"IEND", &[]);
        png
    }

    #[test]
    fn store_evicts_one_lru_entry_at_a_time_without_clearing() {
        // Exercise far more distinct DMI paths than the cache can hold. The old
        // behaviour cleared the entire map at the boundary; bounded LRU eviction
        // must instead drop exactly one (the least-recently-used) entry.
        let mut cache: HashMap<PathBuf, CachedDmiMetadata> = HashMap::new();
        let total = MAX_DMI_METADATA_CACHE_ENTRIES + 500;
        for index in 0..total {
            let path = PathBuf::from(format!("/virtual/dmi/{index}.dmi"));
            store_dmi_metadata(&mut cache, &path, synthetic_entry(index as u64));
        }

        assert_eq!(
            cache.len(),
            MAX_DMI_METADATA_CACHE_ENTRIES,
            "eviction keeps the cache exactly at capacity, never clears it"
        );
        // The 500 oldest keys were evicted one-by-one; everything newer survived.
        assert!(!cache.contains_key(&PathBuf::from("/virtual/dmi/0.dmi")));
        assert!(!cache.contains_key(&PathBuf::from("/virtual/dmi/499.dmi")));
        assert!(cache.contains_key(&PathBuf::from("/virtual/dmi/500.dmi")));
        assert!(cache.contains_key(&PathBuf::from(format!("/virtual/dmi/{}.dmi", total - 1))));

        // Re-touching an existing key must not evict anything.
        let hot = PathBuf::from("/virtual/dmi/500.dmi");
        store_dmi_metadata(&mut cache, &hot, synthetic_entry(u64::MAX));
        assert_eq!(cache.len(), MAX_DMI_METADATA_CACHE_ENTRIES);
        assert!(cache.contains_key(&hot));
    }

    #[test]
    fn repeated_reads_under_the_cap_decode_each_dmi_once() {
        let root = std::env::temp_dir().join(format!(
            "dream64-dmi-cache-bounded-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let mut paths = Vec::new();
        for index in 0..24u32 {
            let path = root.join(format!("sheet-{index}.dmi"));
            std::fs::write(&path, minimal_dmi(16 + index, 32)).unwrap();
            paths.push(path);
        }

        for _ in 0..8 {
            for (index, path) in paths.iter().enumerate() {
                let metadata = read_dmi_metadata(path).unwrap();
                assert_eq!(metadata.width, 16 + index as u32);
            }
        }

        let reads = DMI_METADATA_PHYSICAL_READS.get().unwrap().lock().unwrap();
        for path in &paths {
            assert_eq!(
                reads.get(path).copied(),
                Some(1),
                "each DMI under the cache cap is decoded once across repeated references"
            );
        }
        drop(reads);
        let _ = std::fs::remove_dir_all(&root);
    }
}
